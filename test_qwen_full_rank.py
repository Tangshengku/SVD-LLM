# coding=utf-8
import argparse
import math

import torch
import torch.nn as nn
from tqdm import tqdm

from component.svd_qwen3 import SVD_Qwen3Attention, SVD_Qwen3MLP
from utils.data_utils import get_test_data
from utils.model_utils import get_model_from_huggingface


def qwen_full_rank_ratios(config):
    hidden = config.hidden_size
    intermediate = config.intermediate_size
    attn_ratio = 2.0
    mlp_ratio = (hidden + intermediate) / max(hidden, intermediate)
    return attn_ratio, mlp_ratio


def exact_factorize_linear_for_inner_dim(weight: torch.Tensor, inner_dim: int, factor_device: str):
    out_features, in_features = weight.shape
    full_rank = min(out_features, in_features)
    if inner_dim < full_rank:
        raise ValueError(
            f"Inner dim {inner_dim} is too small for exact factorization of weight shape {tuple(weight.shape)}"
        )

    weight_fp32 = weight.detach().to(device=factor_device, dtype=torch.float32)
    u, s, vh = torch.linalg.svd(weight_fp32, full_matrices=False)
    u = u[:, :full_rank]
    s = s[:full_rank]
    vh = vh[:full_rank, :]
    sqrt_s = torch.sqrt(s)
    left = (u * sqrt_s.unsqueeze(0)).to(dtype=weight.dtype)
    right = (sqrt_s.unsqueeze(1) * vh).to(dtype=weight.dtype)

    padded_left = torch.zeros((out_features, inner_dim), dtype=weight.dtype, device=factor_device)
    padded_right = torch.zeros((inner_dim, in_features), dtype=weight.dtype, device=factor_device)
    padded_left[:, :full_rank] = left
    padded_right[:full_rank, :] = right
    return padded_left.cpu(), padded_right.cpu()


def replace_qwen3_with_full_rank_svd(model, factor_device: str):
    config = model.config
    attn_ratio, mlp_ratio = qwen_full_rank_ratios(config)

    for layer in tqdm(model.model.layers, desc="replacing qwen3 layers with full-rank svd"):
        attn_dtype = layer.self_attn.q_proj.weight.dtype
        mlp_dtype = layer.mlp.gate_proj.weight.dtype

        svd_attn = SVD_Qwen3Attention(
            config=config,
            ratio=attn_ratio,
            init_scheme="default",
            layer_idx=getattr(layer.self_attn, "layer_idx", None),
        ).to(dtype=attn_dtype)
        svd_mlp = SVD_Qwen3MLP(
            config=config,
            ratio=mlp_ratio,
            init_scheme="default",
        ).to(dtype=mlp_dtype)

        for src_name, u_name, v_name in (
            ("q_proj", "q_u_proj", "q_v_proj"),
            ("k_proj", "k_u_proj", "k_v_proj"),
            ("v_proj", "v_u_proj", "v_v_proj"),
            ("o_proj", "o_u_proj", "o_v_proj"),
        ):
            src = getattr(layer.self_attn, src_name)
            inner_dim = getattr(svd_attn, u_name).weight.shape[1]
            left, right = exact_factorize_linear_for_inner_dim(src.weight.data, inner_dim, factor_device)
            getattr(svd_attn, u_name).weight.data.copy_(left)
            getattr(svd_attn, v_name).weight.data.copy_(right)
            if src.bias is not None and getattr(svd_attn, u_name).bias is not None:
                getattr(svd_attn, u_name).bias.data.copy_(src.bias.data)

        for src_name, u_name, v_name in (
            ("gate_proj", "gate_u_proj", "gate_v_proj"),
            ("down_proj", "down_u_proj", "down_v_proj"),
            ("up_proj", "up_u_proj", "up_v_proj"),
        ):
            src = getattr(layer.mlp, src_name)
            inner_dim = getattr(svd_mlp, u_name).weight.shape[1]
            left, right = exact_factorize_linear_for_inner_dim(src.weight.data, inner_dim, factor_device)
            getattr(svd_mlp, u_name).weight.data.copy_(left)
            getattr(svd_mlp, v_name).weight.data.copy_(right)

        svd_attn.q_norm.weight.data.copy_(layer.self_attn.q_norm.weight.data)
        svd_attn.k_norm.weight.data.copy_(layer.self_attn.k_norm.weight.data)

        layer.self_attn = svd_attn
        layer.mlp = svd_mlp


@torch.no_grad()
def collect_logits(model, batches, device):
    logits = []
    model.eval()
    for batch in tqdm(batches, desc="collecting logits"):
        logits.append(model(batch.to(device), use_cache=False).logits.cpu().float())
    return logits


def summarize_logit_deltas(reference_logits, replaced_logits):
    max_abs = 0.0
    mean_abs_total = 0.0
    count = 0
    for ref, new in zip(reference_logits, replaced_logits):
        diff = (new - ref).abs()
        max_abs = max(max_abs, diff.max().item())
        mean_abs_total += diff.mean().item()
        count += 1
    mean_abs = mean_abs_total / max(count, 1)
    return {"max_abs": max_abs, "mean_abs": mean_abs}


@torch.no_grad()
def compute_ppl(model, test_loader, device, max_batches=None):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    loss_fct = nn.CrossEntropyLoss(reduction="sum")

    for batch_idx, batch in enumerate(tqdm(test_loader, desc="computing ppl")):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = batch.to(device)
        logits = model(batch, use_cache=False).logits[:, :-1, :].float()
        labels = batch[:, 1:].contiguous()
        loss = loss_fct(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        total_loss += loss.item()
        total_tokens += labels.numel()

    if total_tokens == 0:
        raise ValueError("No evaluation tokens available for PPL computation")
    return math.exp(total_loss / total_tokens)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--dataset", type=str, default="wikitext2", choices=["wikitext2", "ptb", "c4"])
    parser.add_argument("--model_seq_len", type=int, default=2048)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--compare_batches", type=int, default=2, help="Number of eval batches used for logit comparison.")
    parser.add_argument("--max_eval_batches", type=int, default=8, help="Number of eval batches used for PPL comparison.")
    parser.add_argument("--DEV", type=str, default="cuda")
    parser.add_argument(
        "--factor_device",
        type=str,
        default=None,
        help="Device used for torch.linalg.svd during exact full-rank factorization. Defaults to --DEV.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    factor_device = args.factor_device or args.DEV
    model, tokenizer = get_model_from_huggingface(args.model)
    model.eval()
    model.seqlen = args.model_seq_len
    model.config.use_cache = False
    model = model.to(args.DEV)

    compare_loader = get_test_data(
        args.dataset,
        tokenizer,
        seq_len=args.model_seq_len,
        batch_size=args.eval_batch_size,
    )
    compare_batches = []
    for batch_idx, batch in enumerate(compare_loader):
        if batch_idx >= args.compare_batches:
            break
        compare_batches.append(batch)
    if not compare_batches:
        raise ValueError("No batches available for logit comparison")

    dense_logits = collect_logits(model, compare_batches, args.DEV)

    dense_loader = get_test_data(
        args.dataset,
        tokenizer,
        seq_len=args.model_seq_len,
        batch_size=args.eval_batch_size,
    )
    dense_ppl = compute_ppl(model, dense_loader, args.DEV, max_batches=args.max_eval_batches)

    model = model.cpu()
    torch.cuda.empty_cache()
    replace_qwen3_with_full_rank_svd(model, factor_device)
    model = model.to(args.DEV)
    model.eval()
    model.config.use_cache = False

    replaced_logits = collect_logits(model, compare_batches, args.DEV)
    deltas = summarize_logit_deltas(dense_logits, replaced_logits)

    replaced_loader = get_test_data(
        args.dataset,
        tokenizer,
        seq_len=args.model_seq_len,
        batch_size=args.eval_batch_size,
    )
    replaced_ppl = compute_ppl(model, replaced_loader, args.DEV, max_batches=args.max_eval_batches)

    print(
        {
            "dataset": args.dataset,
            "compare_batches": args.compare_batches,
            "max_eval_batches": args.max_eval_batches,
            "dense_ppl": dense_ppl,
            "svd_full_rank_ppl": replaced_ppl,
            "ppl_delta": replaced_ppl - dense_ppl,
            "max_abs_logit_diff": deltas["max_abs"],
            "mean_abs_logit_diff": deltas["mean_abs"],
        }
    )


if __name__ == "__main__":
    main()
