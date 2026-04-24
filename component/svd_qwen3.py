# coding=utf-8
"""PyTorch Qwen3 low-rank modules."""

import warnings
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
try:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
except ImportError:
    ALL_ATTENTION_FUNCTIONS = None

try:
    from transformers.models.qwen3 import Qwen3Config
except ImportError:
    try:
        from transformers import Qwen3Config
    except ImportError:
        Qwen3Config = object

from component.svd_mistral import (
    MistralRMSNorm,
    SVD_MistralMLP,
    _init_linear,
    apply_rotary_pos_emb,
    repeat_kv,
)


class SVD_Qwen3MLP(SVD_MistralMLP):
    def __init__(self, config: Qwen3Config, ratio=1, init_scheme: str = "uniform"):
        super().__init__(config=config, ratio=ratio, init_scheme=init_scheme)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class SVD_Qwen3Attention(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        ratio=1,
        init_scheme: str = "uniform",
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.init_scheme = init_scheme
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", None) or self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.is_causal = True
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)
        self.ratio = ratio

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got hidden_size={self.hidden_size}, "
                f"num_heads={self.num_heads})."
            )

        low_rank = int(self.hidden_size * self.ratio / 2)
        attention_bias = getattr(config, "attention_bias", False)
        self.q_u_proj = nn.Linear(low_rank, self.num_heads * self.head_dim, bias=attention_bias)
        self.q_v_proj = nn.Linear(self.hidden_size, low_rank, bias=False)
        self.k_u_proj = nn.Linear(low_rank, self.num_key_value_heads * self.head_dim, bias=attention_bias)
        self.k_v_proj = nn.Linear(self.hidden_size, low_rank, bias=False)
        self.v_u_proj = nn.Linear(low_rank, self.num_key_value_heads * self.head_dim, bias=attention_bias)
        self.v_v_proj = nn.Linear(self.hidden_size, low_rank, bias=False)
        self.o_u_proj = nn.Linear(low_rank, self.hidden_size, bias=attention_bias)
        self.o_v_proj = nn.Linear(self.num_heads * self.head_dim, low_rank, bias=False)

        self.q_norm = MistralRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = MistralRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        layer_types = getattr(config, "layer_types", None)
        if layer_types is not None and layer_idx is not None:
            self.layer_type = layer_types[layer_idx]
            self.sliding_window = getattr(config, "sliding_window", None) if self.layer_type == "sliding_attention" else None
        else:
            self.layer_type = "full_attention"
            self.sliding_window = None

        self.reset_parameters()

    def reset_parameters(self):
        for linear in (
            self.q_u_proj,
            self.q_v_proj,
            self.k_u_proj,
            self.k_v_proj,
            self.v_u_proj,
            self.v_v_proj,
            self.o_u_proj,
            self.o_v_proj,
        ):
            _init_linear(linear, self.init_scheme)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[object] = None,
        past_key_value: Optional[object] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in a future version. "
                "Use `attention_mask` instead."
            )
        if past_key_values is None:
            past_key_values = past_key_value

        bsz, q_len, _ = hidden_states.size()
        hidden_shape = (bsz, q_len, -1, self.head_dim)

        query_states = self.q_u_proj(self.q_v_proj(hidden_states)).view(hidden_shape)
        key_states = self.k_u_proj(self.k_v_proj(hidden_states)).view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = self.v_u_proj(self.v_v_proj(hidden_states)).view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        query_states = self.q_norm(query_states).transpose(1, 2)
        key_states = self.k_norm(key_states).transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if position_embeddings is None:
            raise ValueError("Qwen3 attention expects precomputed position_embeddings.")
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        is_legacy_cache = isinstance(past_key_values, tuple)
        if past_key_values is not None and hasattr(past_key_values, "update"):
            cache_kwargs = {"cache_position": cache_position}
            if position_embeddings is not None:
                cache_kwargs["sin"] = sin
                cache_kwargs["cos"] = cos
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        elif is_legacy_cache:
            key_states = torch.cat([past_key_values[0], key_states], dim=2)
            value_states = torch.cat([past_key_values[1], value_states], dim=2)

        present_key_value = (key_states, value_states) if use_cache and is_legacy_cache else None

        attention_interface = eager_attention_forward
        if ALL_ATTENTION_FUNCTIONS is not None and getattr(self.config, "_attn_implementation", "eager") != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_u_proj(self.o_v_proj(attn_output))

        if not output_attentions:
            attn_weights = None

        if hasattr(past_key_values, "update") or position_embeddings is not None or cache_position is not None:
            return attn_output, attn_weights

        return attn_output, attn_weights, present_key_value
