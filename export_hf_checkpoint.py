import argparse
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn

from component.low_rank_linear import LowRankLinear, ZeroLinear
from component.svd_llama import SVD_LlamaAttention, SVD_LlamaMLP
from component.svd_mistral import SVD_MistralAttention, SVD_MistralMLP
from component.svd_opt import SVDOPTAttention, SVDOPTDecoderLayer
from component.svd_qwen3 import SVD_Qwen3Attention, SVD_Qwen3MLP


class ModuleScaffold(nn.Module):
    pass


FLOAT8_DTYPES = tuple(
    dtype
    for dtype in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e5m2fnuz", None),
    )
    if dtype is not None
)


def set_submodule(root: nn.Module, path: str, module: nn.Module) -> None:
    if "." in path:
        parent_path, attr_name = path.rsplit(".", 1)
        parent = root.get_submodule(parent_path)
    else:
        parent = root
        attr_name = path
    setattr(parent, attr_name, module)


def module_device(module: nn.Module) -> torch.device:
    for tensor in module.parameters(recurse=True):
        return tensor.device
    for tensor in module.buffers(recurse=True):
        return tensor.device
    return torch.device("cpu")


def module_dtype(module: nn.Module) -> torch.dtype:
    for tensor in module.parameters(recurse=True):
        if tensor.is_floating_point():
            return tensor.dtype
    for tensor in module.buffers(recurse=True):
        if tensor.is_floating_point():
            return tensor.dtype
    return torch.float16


def clear_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def is_float8_dtype(dtype: torch.dtype) -> bool:
    return dtype in FLOAT8_DTYPES


def make_linear(in_features: int, out_features: int, bias: bool, device: torch.device, dtype: torch.dtype) -> nn.Linear:
    init_dtype = torch.float16 if is_float8_dtype(dtype) else dtype
    dense = nn.Linear(
        in_features,
        out_features,
        bias=bias,
        device=device,
        dtype=init_dtype,
    )
    if is_float8_dtype(dtype):
        dense.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype),
            requires_grad=False,
        )
    return dense


def factorized_linear_to_dense(
    u_proj: nn.Linear,
    v_proj: nn.Linear,
    compute_device: torch.device,
    compute_dtype: torch.dtype,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    u = u_proj.weight.detach().to(device=compute_device, dtype=compute_dtype)
    v = v_proj.weight.detach().to(device=compute_device, dtype=compute_dtype)
    weight = (u @ v).to(dtype=output_dtype).cpu()
    del u, v
    clear_device_cache(compute_device)
    return weight


def low_rank_to_linear(module: LowRankLinear, compute_device: torch.device, compute_dtype: torch.dtype, output_dtype: torch.dtype) -> nn.Linear:
    weight = factorized_linear_to_dense(module.u_proj, module.v_proj, compute_device, compute_dtype, output_dtype)
    bias = module.u_proj.bias is not None
    dense = make_linear(module.in_features, module.out_features, bias, torch.device("cpu"), output_dtype)
    dense.weight.data.copy_(weight)
    if bias:
        dense.bias.data.copy_(module.u_proj.bias.detach().to(device="cpu", dtype=dense.bias.dtype))
    return dense


def zero_to_linear(module: ZeroLinear, dtype: torch.dtype, device: torch.device) -> nn.Linear:
    bias = module.bias is not None
    dense = make_linear(module.in_features, module.out_features, bias, device, dtype)
    dense.weight.data.zero_()
    if bias:
        dense.bias.data.copy_(module.bias.data.to(device=device, dtype=dtype))
    return dense


def convert_factorized_pair(
    u_proj: nn.Linear,
    v_proj: nn.Linear,
    out_features: int,
    in_features: int,
    compute_device: torch.device,
    compute_dtype: torch.dtype,
    output_dtype: torch.dtype,
) -> nn.Linear:
    weight = factorized_linear_to_dense(u_proj, v_proj, compute_device, compute_dtype, output_dtype)
    expected_shape = (out_features, in_features)
    if tuple(weight.shape) != expected_shape:
        raise ValueError(
            f"Factorized pair reconstructed shape {tuple(weight.shape)} does not match "
            f"requested dense shape {expected_shape}. "
            f"u_proj.weight={tuple(u_proj.weight.shape)}, v_proj.weight={tuple(v_proj.weight.shape)}"
        )
    bias = u_proj.bias is not None
    dense = make_linear(in_features, out_features, bias, torch.device("cpu"), output_dtype)
    dense.weight.data.copy_(weight)
    if bias:
        dense.bias.data.copy_(u_proj.bias.detach().to(device="cpu", dtype=dense.bias.dtype))
    return dense


def convert_svd_llama_attention(module: SVD_LlamaAttention, compute_device: torch.device, compute_dtype: torch.dtype, output_dtype: torch.dtype) -> nn.Module:
    dense = ModuleScaffold()
    attention_hidden_size = getattr(module, "attention_hidden_size", module.num_heads * module.head_dim)
    key_value_hidden_size = getattr(
        module,
        "key_value_hidden_size",
        getattr(module, "num_key_value_heads", module.num_heads) * module.head_dim,
    )
    dense.q_proj = convert_factorized_pair(module.q_u_proj, module.q_v_proj, attention_hidden_size, module.hidden_size, compute_device, compute_dtype, output_dtype)
    dense.k_proj = convert_factorized_pair(module.k_u_proj, module.k_v_proj, key_value_hidden_size, module.hidden_size, compute_device, compute_dtype, output_dtype)
    dense.v_proj = convert_factorized_pair(module.v_u_proj, module.v_v_proj, key_value_hidden_size, module.hidden_size, compute_device, compute_dtype, output_dtype)
    dense.o_proj = convert_factorized_pair(module.o_u_proj, module.o_v_proj, module.hidden_size, attention_hidden_size, compute_device, compute_dtype, output_dtype)
    return dense


def convert_svd_mistral_attention(module: SVD_MistralAttention, compute_device: torch.device, compute_dtype: torch.dtype, output_dtype: torch.dtype) -> nn.Module:
    dense = ModuleScaffold()
    dense.layer_idx = getattr(module, "layer_idx", None)
    dense.q_proj = convert_factorized_pair(module.q_u_proj, module.q_v_proj, module.num_heads * module.head_dim, module.hidden_size, compute_device, compute_dtype, output_dtype)
    dense.k_proj = convert_factorized_pair(module.k_u_proj, module.k_v_proj, module.num_key_value_heads * module.head_dim, module.hidden_size, compute_device, compute_dtype, output_dtype)
    dense.v_proj = convert_factorized_pair(module.v_u_proj, module.v_v_proj, module.num_key_value_heads * module.head_dim, module.hidden_size, compute_device, compute_dtype, output_dtype)
    dense.o_proj = convert_factorized_pair(module.o_u_proj, module.o_v_proj, module.hidden_size, module.num_heads * module.head_dim, compute_device, compute_dtype, output_dtype)
    return dense


def convert_svd_qwen3_attention(module: SVD_Qwen3Attention, compute_device: torch.device, compute_dtype: torch.dtype, output_dtype: torch.dtype) -> nn.Module:
    dense = ModuleScaffold()
    dense.layer_idx = getattr(module, "layer_idx", None)
    dense.q_proj = convert_factorized_pair(module.q_u_proj, module.q_v_proj, module.num_heads * module.head_dim, module.hidden_size, compute_device, compute_dtype, output_dtype)
    dense.k_proj = convert_factorized_pair(module.k_u_proj, module.k_v_proj, module.num_key_value_heads * module.head_dim, module.hidden_size, compute_device, compute_dtype, output_dtype)
    dense.v_proj = convert_factorized_pair(module.v_u_proj, module.v_v_proj, module.num_key_value_heads * module.head_dim, module.hidden_size, compute_device, compute_dtype, output_dtype)
    dense.o_proj = convert_factorized_pair(module.o_u_proj, module.o_v_proj, module.hidden_size, module.num_heads * module.head_dim, compute_device, compute_dtype, output_dtype)
    dense.q_norm = module.q_norm
    dense.k_norm = module.k_norm
    return dense


def convert_svd_mlp(module: nn.Module, hidden_size: int, intermediate_size: int, compute_device: torch.device, compute_dtype: torch.dtype, output_dtype: torch.dtype) -> nn.Module:
    dense = ModuleScaffold()
    dense.gate_proj = convert_factorized_pair(module.gate_u_proj, module.gate_v_proj, intermediate_size, hidden_size, compute_device, compute_dtype, output_dtype)
    dense.up_proj = convert_factorized_pair(module.up_u_proj, module.up_v_proj, intermediate_size, hidden_size, compute_device, compute_dtype, output_dtype)
    dense.down_proj = convert_factorized_pair(module.down_u_proj, module.down_v_proj, hidden_size, intermediate_size, compute_device, compute_dtype, output_dtype)
    return dense


def infer_svd_mlp_dims(module: nn.Module) -> Tuple[int, int]:
    hidden_size = getattr(module, "hidden_size", None)
    intermediate_size = getattr(module, "intermediate_size", None)
    if hidden_size is None:
        hidden_size = module.gate_v_proj.in_features
    if intermediate_size is None:
        intermediate_size = module.gate_u_proj.out_features
    return hidden_size, intermediate_size


def convert_svd_opt_attention(module: SVDOPTAttention, compute_device: torch.device, compute_dtype: torch.dtype, output_dtype: torch.dtype) -> nn.Module:
    dense = ModuleScaffold()
    if module.ratio != 1:
        dense.q_proj = convert_factorized_pair(module.q_u_proj, module.q_v_proj, module.embed_dim, module.embed_dim, compute_device, compute_dtype, output_dtype)
        dense.k_proj = convert_factorized_pair(module.k_u_proj, module.k_v_proj, module.embed_dim, module.embed_dim, compute_device, compute_dtype, output_dtype)
        dense.v_proj = convert_factorized_pair(module.v_u_proj, module.v_v_proj, module.embed_dim, module.embed_dim, compute_device, compute_dtype, output_dtype)
        dense.out_proj = convert_factorized_pair(module.out_u_proj, module.out_v_proj, module.embed_dim, module.embed_dim, compute_device, compute_dtype, output_dtype)
    else:
        dense.q_proj = module.q_proj
        dense.k_proj = module.k_proj
        dense.v_proj = module.v_proj
        dense.out_proj = module.out_proj
    return dense


def convert_svd_opt_decoder_layer(module: SVDOPTDecoderLayer, compute_device: torch.device, compute_dtype: torch.dtype, output_dtype: torch.dtype) -> nn.Module:
    dense = ModuleScaffold()
    dense.self_attn = convert_svd_opt_attention(module.self_attn, compute_device, compute_dtype, output_dtype)
    dense.self_attn_layer_norm = module.self_attn_layer_norm
    dense.final_layer_norm = module.final_layer_norm
    if module.ratio != 1:
        dense.fc1 = convert_factorized_pair(module.fc1_u_proj, module.fc1_v_proj, module.fc1_u_proj.out_features, module.fc1_v_proj.in_features, compute_device, compute_dtype, output_dtype)
        dense.fc2 = convert_factorized_pair(module.fc2_u_proj, module.fc2_v_proj, module.fc2_u_proj.out_features, module.fc2_v_proj.in_features, compute_device, compute_dtype, output_dtype)
    else:
        dense.fc1 = module.fc1
        dense.fc2 = module.fc2
    return dense


@torch.no_grad()
def module_replacement_type(module: nn.Module) -> Optional[str]:
    if isinstance(module, (LowRankLinear, ZeroLinear)):
        return "evo"
    if isinstance(
        module,
        (
            SVD_LlamaAttention,
            SVD_MistralAttention,
            SVD_Qwen3Attention,
            SVD_LlamaMLP,
            SVD_MistralMLP,
            SVD_Qwen3MLP,
            SVDOPTDecoderLayer,
            SVDOPTAttention,
        ),
    ):
        return "svdllm"
    return None


def convert_single_module(
    module: nn.Module,
    compute_device: torch.device,
    compute_dtype: torch.dtype,
    output_dtype: torch.dtype,
) -> Tuple[Optional[nn.Module], Optional[str]]:
    if isinstance(module, LowRankLinear):
        return low_rank_to_linear(module, compute_device, compute_dtype, output_dtype), "evo"
    if isinstance(module, ZeroLinear):
        return zero_to_linear(module, output_dtype, torch.device("cpu")), "evo"
    if isinstance(module, SVD_LlamaAttention):
        return convert_svd_llama_attention(module, compute_device, compute_dtype, output_dtype), "svdllm"
    if isinstance(module, SVD_MistralAttention):
        return convert_svd_mistral_attention(module, compute_device, compute_dtype, output_dtype), "svdllm"
    if isinstance(module, SVD_Qwen3Attention):
        return convert_svd_qwen3_attention(module, compute_device, compute_dtype, output_dtype), "svdllm"
    if isinstance(module, SVD_LlamaMLP):
        hidden_size, intermediate_size = infer_svd_mlp_dims(module)
        return convert_svd_mlp(module, hidden_size, intermediate_size, compute_device, compute_dtype, output_dtype), "svdllm"
    if isinstance(module, SVD_MistralMLP):
        hidden_size, intermediate_size = infer_svd_mlp_dims(module)
        return convert_svd_mlp(module, hidden_size, intermediate_size, compute_device, compute_dtype, output_dtype), "svdllm"
    if isinstance(module, SVD_Qwen3MLP):
        hidden_size, intermediate_size = infer_svd_mlp_dims(module)
        return convert_svd_mlp(module, hidden_size, intermediate_size, compute_device, compute_dtype, output_dtype), "svdllm"
    if isinstance(module, SVDOPTDecoderLayer):
        return convert_svd_opt_decoder_layer(module, compute_device, compute_dtype, output_dtype), "svdllm"
    if isinstance(module, SVDOPTAttention):
        return convert_svd_opt_attention(module, compute_device, compute_dtype, output_dtype), "svdllm"
    return None, None


@torch.no_grad()
def convert_modules(
    model: nn.Module,
    compute_device: torch.device,
    compute_dtype: torch.dtype,
    output_dtype: torch.dtype,
) -> tuple[int, int]:
    replacement_names = []
    evo_replacements = 0
    svdllm_replacements = 0

    for name, module in model.named_modules():
        if module_replacement_type(module) is not None:
            replacement_names.append(name)

    for idx, name in enumerate(replacement_names):
        module = model.get_submodule(name)
        converted, replacement_type = convert_single_module(module, compute_device, compute_dtype, output_dtype)
        if converted is None:
            continue
        set_submodule(model, name, converted)
        if replacement_type == "evo":
            evo_replacements += 1
        else:
            svdllm_replacements += 1
        del module, converted
        clear_device_cache(compute_device)
        if (idx + 1) % 20 == 0 or idx + 1 == len(replacement_names):
            print(f"Converted {idx + 1}/{len(replacement_names)} low-rank modules", flush=True)

    return evo_replacements, svdllm_replacements


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a saved SVD-LLM .pt checkpoint to Hugging Face Transformers format."
    )
    parser.add_argument("--input_pt", type=str, required=True, help="Path to the saved .pt checkpoint.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the HF checkpoint.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device used only for one dense weight reconstruction at a time.",
    )
    parser.add_argument(
        "--compute_dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Matmul dtype used on --device before casting each dense weight to --output_dtype.",
    )
    parser.add_argument(
        "--output_dtype",
        choices=["float8_e4m3fn", "float8_e5m2", "float16", "bfloat16", "float32"],
        default="float8_e4m3fn",
        help="Dense exported weight dtype. Use float8_e4m3fn to keep FP8-style exported weights.",
    )
    parser.add_argument(
        "--max_shard_size",
        type=str,
        default="2GB",
        help="Maximum shard size passed to save_pretrained.",
    )
    parser.add_argument(
        "--save_safetensors",
        action="store_true",
        help="Save weights as safetensors instead of pytorch_model.bin.",
    )
    return parser.parse_args()


def parse_output_dtype(name: str) -> torch.dtype:
    if name == "float8_e4m3fn":
        if not hasattr(torch, "float8_e4m3fn"):
            raise ValueError("This PyTorch build does not expose torch.float8_e4m3fn")
        return torch.float8_e4m3fn
    if name == "float8_e5m2":
        if not hasattr(torch, "float8_e5m2"):
            raise ValueError("This PyTorch build does not expose torch.float8_e5m2")
        return torch.float8_e5m2
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def main():
    args = parse_args()
    checkpoint = torch.load(args.input_pt, map_location="cpu", weights_only=False)

    if "model" not in checkpoint or "tokenizer" not in checkpoint:
        raise ValueError("Expected checkpoint format {'model': ..., 'tokenizer': ...}.")

    model = checkpoint["model"].cpu()
    tokenizer = checkpoint["tokenizer"]
    model.eval()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device '{args.device}' but CUDA is not available.")

    compute_device = torch.device(args.device)
    compute_dtype = parse_output_dtype(args.compute_dtype)
    output_dtype = parse_output_dtype(args.output_dtype)

    evo_replacements, svdllm_replacements = convert_modules(model, compute_device, compute_dtype, output_dtype)
    os.makedirs(args.output_dir, exist_ok=True)

    model = model.cpu()
    model.save_pretrained(
        args.output_dir,
        safe_serialization=args.save_safetensors,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.save_pretrained(args.output_dir)

    print(f"Converted {evo_replacements} evo_svdllm low-rank linear modules on {compute_device}.")
    print(f"Converted {svdllm_replacements} SVDLLM structural low-rank modules on {compute_device}.")
    print(
        f"Exported dense weights with compute_dtype={compute_dtype}, "
        f"output_dtype={output_dtype}, and max_shard_size={args.max_shard_size}."
    )
    print(f"Saved Hugging Face checkpoint to: {args.output_dir}")


if __name__ == "__main__":
    main()
