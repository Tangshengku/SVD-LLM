import argparse
import os

import torch
import torch.nn as nn

from component.low_rank_linear import LowRankLinear, ZeroLinear
from component.svd_llama import SVD_LlamaAttention, SVD_LlamaMLP
from component.svd_mistral import SVD_MistralAttention, SVD_MistralMLP
from component.svd_opt import SVDOPTAttention, SVDOPTDecoderLayer
from component.svd_qwen3 import SVD_Qwen3Attention, SVD_Qwen3MLP


class ModuleScaffold(nn.Module):
    pass


def set_submodule(root: nn.Module, path: str, module: nn.Module) -> None:
    if "." in path:
        parent_path, attr_name = path.rsplit(".", 1)
        parent = root.get_submodule(parent_path)
    else:
        parent = root
        attr_name = path
    setattr(parent, attr_name, module)


def make_linear(in_features: int, out_features: int, bias: bool, device: torch.device, dtype: torch.dtype) -> nn.Linear:
    return nn.Linear(
        in_features,
        out_features,
        bias=bias,
        device=device,
        dtype=dtype,
    )


def factorized_linear_to_dense(u_proj: nn.Linear, v_proj: nn.Linear) -> tuple[torch.Tensor, torch.dtype, torch.device]:
    weight = u_proj.weight.data @ v_proj.weight.data
    return weight, u_proj.weight.dtype, u_proj.weight.device


def low_rank_to_linear(module: LowRankLinear) -> nn.Linear:
    weight, dtype, device = factorized_linear_to_dense(module.u_proj, module.v_proj)
    bias = module.u_proj.bias is not None
    dense = make_linear(module.in_features, module.out_features, bias, device, dtype)
    dense.weight.data.copy_(weight)
    if bias:
        dense.bias.data.copy_(module.u_proj.bias.data)
    return dense


def zero_to_linear(module: ZeroLinear, dtype: torch.dtype, device: torch.device) -> nn.Linear:
    bias = module.bias is not None
    dense = make_linear(module.in_features, module.out_features, bias, device, dtype)
    dense.weight.data.zero_()
    if bias:
        dense.bias.data.copy_(module.bias.data.to(device=device, dtype=dtype))
    return dense


def convert_factorized_pair(u_proj: nn.Linear, v_proj: nn.Linear, out_features: int, in_features: int) -> nn.Linear:
    weight, dtype, device = factorized_linear_to_dense(u_proj, v_proj)
    bias = u_proj.bias is not None
    dense = make_linear(in_features, out_features, bias, device, dtype)
    dense.weight.data.copy_(weight)
    if bias:
        dense.bias.data.copy_(u_proj.bias.data)
    return dense


def convert_svd_llama_attention(module: SVD_LlamaAttention) -> nn.Module:
    dense = ModuleScaffold()
    dense.q_proj = convert_factorized_pair(module.q_u_proj, module.q_v_proj, module.num_heads * module.head_dim, module.hidden_size)
    dense.k_proj = convert_factorized_pair(module.k_u_proj, module.k_v_proj, module.num_heads * module.head_dim, module.hidden_size)
    dense.v_proj = convert_factorized_pair(module.v_u_proj, module.v_v_proj, module.num_heads * module.head_dim, module.hidden_size)
    dense.o_proj = convert_factorized_pair(module.o_u_proj, module.o_v_proj, module.hidden_size, module.num_heads * module.head_dim)
    return dense


def convert_svd_mistral_attention(module: SVD_MistralAttention) -> nn.Module:
    dense = ModuleScaffold()
    dense.layer_idx = getattr(module, "layer_idx", None)
    dense.q_proj = convert_factorized_pair(module.q_u_proj, module.q_v_proj, module.num_heads * module.head_dim, module.hidden_size)
    dense.k_proj = convert_factorized_pair(module.k_u_proj, module.k_v_proj, module.num_key_value_heads * module.head_dim, module.hidden_size)
    dense.v_proj = convert_factorized_pair(module.v_u_proj, module.v_v_proj, module.num_key_value_heads * module.head_dim, module.hidden_size)
    dense.o_proj = convert_factorized_pair(module.o_u_proj, module.o_v_proj, module.hidden_size, module.num_heads * module.head_dim)
    return dense


def convert_svd_qwen3_attention(module: SVD_Qwen3Attention) -> nn.Module:
    dense = ModuleScaffold()
    dense.layer_idx = getattr(module, "layer_idx", None)
    dense.q_proj = convert_factorized_pair(module.q_u_proj, module.q_v_proj, module.num_heads * module.head_dim, module.hidden_size)
    dense.k_proj = convert_factorized_pair(module.k_u_proj, module.k_v_proj, module.num_key_value_heads * module.head_dim, module.hidden_size)
    dense.v_proj = convert_factorized_pair(module.v_u_proj, module.v_v_proj, module.num_key_value_heads * module.head_dim, module.hidden_size)
    dense.o_proj = convert_factorized_pair(module.o_u_proj, module.o_v_proj, module.hidden_size, module.num_heads * module.head_dim)
    dense.q_norm = module.q_norm
    dense.k_norm = module.k_norm
    return dense


def convert_svd_mlp(module: nn.Module, hidden_size: int, intermediate_size: int) -> nn.Module:
    dense = ModuleScaffold()
    dense.gate_proj = convert_factorized_pair(module.gate_u_proj, module.gate_v_proj, intermediate_size, hidden_size)
    dense.up_proj = convert_factorized_pair(module.up_u_proj, module.up_v_proj, intermediate_size, hidden_size)
    dense.down_proj = convert_factorized_pair(module.down_u_proj, module.down_v_proj, hidden_size, intermediate_size)
    return dense


def convert_svd_opt_attention(module: SVDOPTAttention) -> nn.Module:
    dense = ModuleScaffold()
    if module.ratio != 1:
        dense.q_proj = convert_factorized_pair(module.q_u_proj, module.q_v_proj, module.embed_dim, module.embed_dim)
        dense.k_proj = convert_factorized_pair(module.k_u_proj, module.k_v_proj, module.embed_dim, module.embed_dim)
        dense.v_proj = convert_factorized_pair(module.v_u_proj, module.v_v_proj, module.embed_dim, module.embed_dim)
        dense.out_proj = convert_factorized_pair(module.out_u_proj, module.out_v_proj, module.embed_dim, module.embed_dim)
    else:
        dense.q_proj = module.q_proj
        dense.k_proj = module.k_proj
        dense.v_proj = module.v_proj
        dense.out_proj = module.out_proj
    return dense


def convert_svd_opt_decoder_layer(module: SVDOPTDecoderLayer) -> nn.Module:
    dense = ModuleScaffold()
    dense.self_attn = convert_svd_opt_attention(module.self_attn)
    dense.self_attn_layer_norm = module.self_attn_layer_norm
    dense.final_layer_norm = module.final_layer_norm
    if module.ratio != 1:
        dense.fc1 = convert_factorized_pair(module.fc1_u_proj, module.fc1_v_proj, module.fc1_u_proj.out_features, module.fc1_v_proj.in_features)
        dense.fc2 = convert_factorized_pair(module.fc2_u_proj, module.fc2_v_proj, module.fc2_u_proj.out_features, module.fc2_v_proj.in_features)
    else:
        dense.fc1 = module.fc1
        dense.fc2 = module.fc2
    return dense


@torch.no_grad()
def convert_modules(model: nn.Module) -> tuple[int, int]:
    replacements = []
    fallback_param = next(model.parameters())
    fallback_dtype = fallback_param.dtype
    fallback_device = fallback_param.device
    evo_replacements = 0
    svdllm_replacements = 0

    for name, module in model.named_modules():
        if isinstance(module, LowRankLinear):
            replacements.append((name, low_rank_to_linear(module)))
            evo_replacements += 1
        elif isinstance(module, ZeroLinear):
            replacements.append((name, zero_to_linear(module, fallback_dtype, fallback_device)))
            evo_replacements += 1
        elif isinstance(module, SVD_LlamaAttention):
            replacements.append((name, convert_svd_llama_attention(module)))
            svdllm_replacements += 1
        elif isinstance(module, SVD_MistralAttention):
            replacements.append((name, convert_svd_mistral_attention(module)))
            svdllm_replacements += 1
        elif isinstance(module, SVD_Qwen3Attention):
            replacements.append((name, convert_svd_qwen3_attention(module)))
            svdllm_replacements += 1
        elif isinstance(module, SVD_LlamaMLP):
            replacements.append((name, convert_svd_mlp(module, module.hidden_size, module.intermediate_size)))
            svdllm_replacements += 1
        elif isinstance(module, SVD_MistralMLP):
            replacements.append((name, convert_svd_mlp(module, module.hidden_size, module.intermediate_size)))
            svdllm_replacements += 1
        elif isinstance(module, SVD_Qwen3MLP):
            replacements.append((name, convert_svd_mlp(module, module.hidden_size, module.intermediate_size)))
            svdllm_replacements += 1
        elif isinstance(module, SVDOPTDecoderLayer):
            replacements.append((name, convert_svd_opt_decoder_layer(module)))
            svdllm_replacements += 1
        elif isinstance(module, SVDOPTAttention):
            replacements.append((name, convert_svd_opt_attention(module)))
            svdllm_replacements += 1

    for name, module in replacements:
        set_submodule(model, name, module)

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
        default="cuda",
        help="Device used for reconstructing dense weights, e.g. cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--save_safetensors",
        action="store_true",
        help="Save weights as safetensors instead of pytorch_model.bin.",
    )
    return parser.parse_args()


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
    model = model.to(compute_device)

    evo_replacements, svdllm_replacements = convert_modules(model)
    os.makedirs(args.output_dir, exist_ok=True)

    model = model.cpu()
    model.save_pretrained(args.output_dir, safe_serialization=args.save_safetensors)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Converted {evo_replacements} evo_svdllm low-rank linear modules on {compute_device}.")
    print(f"Converted {svdllm_replacements} SVDLLM structural low-rank modules on {compute_device}.")
    print(f"Saved Hugging Face checkpoint to: {args.output_dir}")


if __name__ == "__main__":
    main()
