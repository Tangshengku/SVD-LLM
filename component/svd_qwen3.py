# coding=utf-8
"""PyTorch Qwen3 low-rank modules.

Qwen3 decoder blocks are close enough to the current Mistral integration in
this repository that we can reuse the same low-rank attention/MLP logic while
keeping a separate module for clarity and future Qwen-specific changes.
"""

from component.svd_mistral import SVD_MistralAttention, SVD_MistralMLP

try:
    from transformers.models.qwen3 import Qwen3Config
except ImportError:
    try:
        from transformers import Qwen3Config
    except ImportError:
        Qwen3Config = object


class SVD_Qwen3MLP(SVD_MistralMLP):
    def __init__(self, config: Qwen3Config, ratio=1, init_scheme: str = "uniform"):
        super().__init__(config=config, ratio=ratio, init_scheme=init_scheme)


class SVD_Qwen3Attention(SVD_MistralAttention):
    def __init__(self, config: Qwen3Config, ratio=1, init_scheme: str = "uniform", layer_idx=None):
        super().__init__(
            config=config,
            ratio=ratio,
            init_scheme=init_scheme,
            layer_idx=layer_idx,
        )
