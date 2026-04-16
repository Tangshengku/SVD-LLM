import argparse
import os

import torch
import torch.nn as nn

from component.low_rank_linear import LowRankLinear, ZeroLinear


def set_submodule(root: nn.Module, path: str, module: nn.Module) -> None:
    if "." in path:
        parent_path, attr_name = path.rsplit(".", 1)
        parent = root.get_submodule(parent_path)
    else:
        parent = root
        attr_name = path
    setattr(parent, attr_name, module)


def low_rank_to_linear(module: LowRankLinear) -> nn.Linear:
    device = module.u_proj.weight.device
    dtype = module.u_proj.weight.dtype
    bias = module.u_proj.bias is not None

    dense = nn.Linear(
        module.in_features,
        module.out_features,
        bias=bias,
        device=device,
        dtype=dtype,
    )
    dense.weight.data.copy_(module.u_proj.weight.data @ module.v_proj.weight.data)
    if bias:
        dense.bias.data.copy_(module.u_proj.bias.data)
    return dense


def zero_to_linear(module: ZeroLinear, dtype: torch.dtype, device: torch.device) -> nn.Linear:
    bias = module.bias is not None
    dense = nn.Linear(
        module.in_features,
        module.out_features,
        bias=bias,
        device=device,
        dtype=dtype,
    )
    dense.weight.data.zero_()
    if bias:
        dense.bias.data.copy_(module.bias.data.to(device=device, dtype=dtype))
    return dense


@torch.no_grad()
def convert_modules(model: nn.Module) -> int:
    replacements = []
    fallback_param = next(model.parameters())
    fallback_dtype = fallback_param.dtype
    fallback_device = fallback_param.device

    for name, module in model.named_modules():
        if isinstance(module, LowRankLinear):
            replacements.append((name, low_rank_to_linear(module)))
        elif isinstance(module, ZeroLinear):
            replacements.append((name, zero_to_linear(module, fallback_dtype, fallback_device)))

    for name, module in replacements:
        set_submodule(model, name, module)

    return len(replacements)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export an evolutionary SVD .pt checkpoint to Hugging Face Transformers format."
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

    converted = convert_modules(model)
    os.makedirs(args.output_dir, exist_ok=True)

    model = model.cpu()
    model.save_pretrained(args.output_dir, safe_serialization=args.save_safetensors)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Converted {converted} low-rank modules to dense nn.Linear layers on {compute_device}.")
    print(f"Saved Hugging Face checkpoint to: {args.output_dir}")


if __name__ == "__main__":
    main()
