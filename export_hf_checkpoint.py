import argparse
import os

import torch

from utils.model_utils import load_dense_model_from_compressed_checkpoint


def parse_dtype(dtype_name):
    dtype_name = dtype_name.lower()
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Local compressed .pt checkpoint.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the exported Hugging Face model.")
    parser.add_argument("--base_model_id", type=str, default=None, help="Optional original Hugging Face base model id.")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"], help="Torch dtype used when rebuilding the dense model.")
    parser.add_argument("--safe_serialization", action="store_true", help="Save as safetensors when possible.")
    parser.add_argument("--max_shard_size", type=str, default="10GB", help="Maximum shard size for save_pretrained.")
    args = parser.parse_args()

    torch_dtype = parse_dtype(args.dtype)
    model, tokenizer = load_dense_model_from_compressed_checkpoint(
        args.checkpoint_path,
        base_model_id=args.base_model_id,
        device_map="cpu",
        torch_dtype=torch_dtype,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(
        args.output_dir,
        safe_serialization=args.safe_serialization,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.save_pretrained(args.output_dir)
    print(f"Exported Hugging Face model to {args.output_dir}")
