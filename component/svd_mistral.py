# coding=utf-8
# Copyright 2023 Mistral AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Mistral low-rank modules compatible with recent Transformers releases."""
import math
import warnings
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from transformers.activations import ACT2FN
try:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
except ImportError:
    ALL_ATTENTION_FUNCTIONS = None
from transformers.utils import (
    logging,
)
try:
    from transformers.models.mistral import MistralConfig
except ImportError:
    from transformers import MistralConfig
try:
    from transformers.models.mistral.modeling_mistral import (
        apply_rotary_pos_emb as hf_apply_rotary_pos_emb,
        eager_attention_forward as hf_eager_attention_forward,
        repeat_kv as hf_repeat_kv,
    )
except ImportError:
    hf_apply_rotary_pos_emb = None
    hf_eager_attention_forward = None
    hf_repeat_kv = None


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "MistralConfig"


def _init_linear(linear: nn.Linear, init_scheme: str) -> None:
    if init_scheme == "uniform":
        bound = 1 / math.sqrt(linear.in_features) if linear.in_features > 0 else 0.0
        nn.init.uniform_(linear.weight, -bound, bound)
        if linear.bias is not None:
            nn.init.uniform_(linear.bias, -bound, bound)
    elif init_scheme == "xavier_uniform":
        nn.init.xavier_uniform_(linear.weight)
        if linear.bias is not None:
            nn.init.zeros_(linear.bias)
    elif init_scheme == "kaiming_uniform":
        nn.init.kaiming_uniform_(linear.weight, a=math.sqrt(5))
        if linear.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(linear.bias, -bound, bound)
    elif init_scheme == "default":
        linear.reset_parameters()
    else:
        raise ValueError(f"Unsupported init_scheme: {init_scheme}")


# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Mistral
class MistralRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        MistralRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding with Llama->Mistral
class MistralRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Apply RoPE to q/k while accepting both old and new Transformers tensor layouts."""
    if hf_apply_rotary_pos_emb is not None and cos.dim() == 3:
        return hf_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=unsqueeze_dim)

    if position_ids is not None:
        cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    else:
        if cos.dim() == 2:
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
        elif cos.dim() == 3:
            cos = cos.unsqueeze(unsqueeze_dim)
            sin = sin.unsqueeze(unsqueeze_dim)
        elif cos.dim() != 4:
            raise ValueError(f"Unsupported RoPE tensor rank: cos={cos.dim()}, sin={sin.dim()}")

        if cos.shape[-2] != q.shape[-2]:
            cos = cos[..., -q.shape[-2] :, :]
            sin = sin[..., -q.shape[-2] :, :]

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SVD_MistralMLP(nn.Module):
    def __init__(self, config,
                 ratio=1,  # 1 means no truncate, just keep normal MLP
                 init_scheme: str = "uniform",
                 ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.ratio = ratio
        self.init_scheme = init_scheme
        low_rank = int(self.intermediate_size * self.hidden_size * self.ratio / (self.intermediate_size + self.hidden_size))
        self.gate_u_proj = nn.Linear(low_rank, self.intermediate_size, bias=False)
        self.gate_v_proj = nn.Linear(self.hidden_size, low_rank, bias=False)

        self.down_u_proj = nn.Linear(low_rank, self.hidden_size, bias=False)
        self.down_v_proj = nn.Linear(self.intermediate_size, low_rank, bias=False)

        self.up_u_proj = nn.Linear(low_rank, self.intermediate_size, bias=False)
        self.up_v_proj = nn.Linear(self.hidden_size, low_rank, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        self.reset_parameters()

    def reset_parameters(self):
        for linear in (
            self.gate_u_proj,
            self.gate_v_proj,
            self.down_u_proj,
            self.down_v_proj,
            self.up_u_proj,
            self.up_v_proj,
        ):
            _init_linear(linear, self.init_scheme)

    def forward(self, x):
        up = self.up_u_proj(self.up_v_proj(x))
        gate = self.gate_u_proj(self.gate_v_proj(x))
        return self.down_u_proj(self.down_v_proj(self.act_fn(gate) * up))


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    if hf_repeat_kv is not None:
        return hf_repeat_kv(hidden_states, n_rep)

    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


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
    if hf_eager_attention_forward is not None:
        return hf_eager_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            scaling=scaling,
            dropout=dropout,
            **kwargs,
        )

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class SVD_MistralAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(
        self,
        config: MistralConfig,
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
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.scaling = self.head_dim ** -0.5
        self.is_causal = True
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)
        self.ratio = ratio # 1 means no truncate, just keep normal attn

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        low_rank = int(self.hidden_size * self.ratio/2)
        self.q_u_proj = nn.Linear(low_rank, self.num_heads * self.head_dim, bias=False)
        self.q_v_proj = nn.Linear(self.hidden_size, low_rank, bias=False)
        self.k_u_proj = nn.Linear(low_rank, self.num_key_value_heads * self.head_dim, bias=False)
        self.k_v_proj = nn.Linear(self.hidden_size, low_rank, bias=False)
        self.v_u_proj = nn.Linear(low_rank, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_v_proj = nn.Linear(self.hidden_size, low_rank, bias=False)
        self.o_u_proj = nn.Linear(low_rank, self.hidden_size, bias=False)
        self.o_v_proj = nn.Linear(self.num_heads * self.head_dim, low_rank, bias=False)

        self.rotary_emb = MistralRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
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

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        if past_key_values is None:
            past_key_values = past_key_value
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_u_proj(self.q_v_proj(hidden_states))
        key_states = self.k_u_proj(self.k_v_proj(hidden_states))
        value_states = self.v_u_proj(self.v_v_proj(hidden_states))

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        is_legacy_cache = isinstance(past_key_values, tuple)
        if is_legacy_cache and past_key_values is not None:
            kv_seq_len += past_key_values[0].shape[-2]
        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_values is not None and hasattr(past_key_values, "update"):
            cache_kwargs = {"cache_position": cache_position}
            if position_embeddings is not None:
                cache_kwargs["sin"] = sin
                cache_kwargs["cos"] = cos
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
            kv_seq_len = key_states.shape[-2]
        elif is_legacy_cache:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_values[0], key_states], dim=2)
            value_states = torch.cat([past_key_values[1], value_states], dim=2)
            kv_seq_len = key_states.shape[-2]

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
            sliding_window=getattr(self.config, "sliding_window", None),
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()

        attn_output = self.o_u_proj(self.o_v_proj(attn_output))

        if not output_attentions:
            attn_weights = None

        if hasattr(past_key_values, "update") or position_embeddings is not None or cache_position is not None:
            return attn_output, attn_weights

        return attn_output, attn_weights, present_key_value
