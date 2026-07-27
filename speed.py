#!/usr/bin/env python3
"""Benchmark vLLM evaluation runtime with and without Qwen MTP.

This script measures generation wall time for HumanEval and GSM8K prompts using
Qwen/Qwen3.5-35B-A3B. It does not execute HumanEval solutions or score GSM8K
answers by default; the goal is to compare the runtime of the same evaluation
workload with normal decoding and native MTP speculative decoding.

Example:
    python speed.py --limit 164 --tensor-parallel-size 4

    python speed.py \
      --model Qwen/Qwen3.5-35B-A3B \
      --datasets humaneval gsm8k \
      --limit 200 \
      --max-tokens 512 \
      --tensor-parallel-size 4 \
      --num-speculative-tokens 1

If vLLM appears to pause after "torch.compile took ...", keep the default
--enforce-eager setting. Add --compile if you explicitly want vLLM compilation
included in engine setup.
"""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


MODEL_NAME = "Qwen/Qwen3.5-35B-A3B"


@dataclass
class BenchmarkResult:
    dataset: str
    mode: str
    model: str
    num_prompts: int
    max_tokens: int
    load_seconds: float
    warmup_seconds: float
    generation_seconds: float
    total_seconds: float
    prompt_tokens: int
    generated_tokens: int
    prompt_tokens_per_second: float
    generated_tokens_per_second: float
    requests_per_second: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare vLLM evaluation runtime with and without MTP."
    )
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["humaneval", "gsm8k"],
        choices=["humaneval", "gsm8k"],
        help="Datasets to benchmark.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum examples per dataset. Default uses the full split.",
    )
    parser.add_argument(
        "--warmup-prompts",
        type=int,
        default=4,
        help="Number of prompts to generate before timing.",
    )
    parser.add_argument("--max-tokens", "--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", "--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument(
        "--max-model-len",
        "--max_model_len",
        type=int,
        default=4096,
        help=(
            "Text evaluation context length. Keeping this below the model default "
            "avoids large vLLM profiling/compile ranges for HumanEval and GSM8K."
        ),
    )
    parser.add_argument(
        "--max-num-seqs",
        "--max_num_seqs",
        type=int,
        default=64,
        help="Maximum concurrent sequences scheduled by vLLM.",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        "--max_num_batched_tokens",
        type=int,
        default=8192,
        help="Maximum tokens per vLLM batch.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        "--gpu_memory_utilization",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--tensor-parallel-size",
        "--tensor_parallel_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--pipeline-parallel-size",
        "--pipeline_parallel_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--disable-log-stats",
        "--disable_log_stats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass disable_log_stats to vLLM to reduce background stat logs.",
    )
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Disable vLLM torch.compile/CUDA graph setup. This is on by default "
            "because the benchmark times generation, not engine compilation."
        ),
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Shortcut for --no-enforce-eager.",
    )
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument(
        "--no-trust-remote-code",
        dest="trust_remote_code",
        action="store_false",
    )
    parser.add_argument(
        "--num-speculative-tokens",
        "--num_speculative_tokens",
        type=int,
        default=1,
        help="MTP speculative depth. vLLM docs recommend starting at 1.",
    )
    parser.add_argument(
        "--mtp-method",
        "--mtp_method",
        default="mtp",
        help=(
            "vLLM speculative method. Use 'mtp' for native MTP on current vLLM; "
            "override if your vLLM build expects a Qwen-specific method name."
        ),
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["no_mtp", "mtp"],
        choices=["no_mtp", "mtp"],
        help="Which decoding modes to run.",
    )
    parser.add_argument(
        "--dataset-cache-dir",
        "--dataset_cache_dir",
        default=None,
        help="Optional Hugging Face datasets cache directory.",
    )
    parser.add_argument(
        "--output",
        default="speed_results.jsonl",
        help="Write one JSON result per benchmark to this path.",
    )
    parser.add_argument(
        "--print-generations",
        action="store_true",
        help="Print generated text samples. Disabled by default for clean timing.",
    )
    return parser.parse_args()


def load_prompts(
    dataset_name: str,
    limit: int | None,
    cache_dir: str | None,
) -> list[str]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install with `pip install datasets`."
        ) from exc

    if dataset_name == "humaneval":
        dataset = load_dataset("openai/openai_humaneval", split="test", cache_dir=cache_dir)
        prompts = [make_humaneval_prompt(row["prompt"]) for row in dataset]
    elif dataset_name == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main", split="test", cache_dir=cache_dir)
        prompts = [make_gsm8k_prompt(row["question"]) for row in dataset]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if limit is not None:
        prompts = prompts[:limit]
    if not prompts:
        raise ValueError(f"No prompts loaded for {dataset_name}")
    return prompts


def make_humaneval_prompt(prompt: str) -> str:
    return (
        "Complete the following Python function. Return only valid Python code.\n\n"
        f"{prompt}"
    )


def make_gsm8k_prompt(question: str) -> str:
    return (
        "Solve the math problem. Show concise reasoning and put the final numeric "
        "answer after '####'.\n\n"
        f"Question: {question}\nAnswer:"
    )


def llm_kwargs(args: argparse.Namespace, mode: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": args.model,
        "trust_remote_code": args.trust_remote_code,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "enforce_eager": args.enforce_eager and not args.compile,
        "disable_log_stats": args.disable_log_stats,
        "seed": args.seed,
    }
    if args.max_model_len is not None:
        kwargs["max_model_len"] = args.max_model_len
    if mode == "mtp":
        kwargs["speculative_config"] = {
            "method": args.mtp_method,
            "num_speculative_tokens": args.num_speculative_tokens,
        }
    return kwargs


def sampling_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
    }
    if args.temperature == 0:
        kwargs["top_p"] = 1.0
    return kwargs


def run_benchmark_child(
    args_dict: dict[str, Any],
    dataset_name: str,
    mode: str,
    prompts: list[str],
    result_queue: mp.Queue,
) -> None:
    args = argparse.Namespace(**args_dict)
    start_total = time.perf_counter()

    try:
        from vllm import LLM, SamplingParams

        start_load = time.perf_counter()
        effective_llm_kwargs = llm_kwargs(args, mode)
        print(
            "[speed.py] LLM kwargs: "
            + json.dumps(effective_llm_kwargs, sort_keys=True),
            flush=True,
        )
        llm = LLM(**effective_llm_kwargs)
        load_seconds = time.perf_counter() - start_load

        sampling_params = SamplingParams(**sampling_kwargs(args))

        warmup_seconds = 0.0
        if args.warmup_prompts > 0:
            warmup = prompts[: args.warmup_prompts]
            start_warmup = time.perf_counter()
            llm.generate(warmup, sampling_params, use_tqdm=False)
            warmup_seconds = time.perf_counter() - start_warmup

        start_generation = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
        generation_seconds = time.perf_counter() - start_generation

        prompt_tokens = sum(count_prompt_tokens(output) for output in outputs)
        generated_tokens = sum(count_generated_tokens(output) for output in outputs)

        if args.print_generations:
            print_generation_samples(dataset_name, mode, outputs)

        total_seconds = time.perf_counter() - start_total
        result = BenchmarkResult(
            dataset=dataset_name,
            mode=mode,
            model=args.model,
            num_prompts=len(prompts),
            max_tokens=args.max_tokens,
            load_seconds=load_seconds,
            warmup_seconds=warmup_seconds,
            generation_seconds=generation_seconds,
            total_seconds=total_seconds,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            prompt_tokens_per_second=safe_rate(prompt_tokens, generation_seconds),
            generated_tokens_per_second=safe_rate(generated_tokens, generation_seconds),
            requests_per_second=safe_rate(len(prompts), generation_seconds),
        )
        result_queue.put({"ok": True, "result": asdict(result)})
    except BaseException as exc:
        result_queue.put(
            {
                "ok": False,
                "dataset": dataset_name,
                "mode": mode,
                "error": repr(exc),
            }
        )
        raise
    finally:
        gc.collect()


def count_prompt_tokens(output: Any) -> int:
    token_ids = getattr(output, "prompt_token_ids", None)
    return len(token_ids) if token_ids is not None else 0


def count_generated_tokens(output: Any) -> int:
    total = 0
    for completion in getattr(output, "outputs", []):
        token_ids = getattr(completion, "token_ids", None)
        if token_ids is not None:
            total += len(token_ids)
    return total


def safe_rate(numerator: int | float, seconds: float) -> float:
    return numerator / seconds if seconds > 0 else 0.0


def print_generation_samples(dataset_name: str, mode: str, outputs: Iterable[Any]) -> None:
    print(f"\n===== samples: {dataset_name} / {mode} =====")
    for i, output in enumerate(list(outputs)[:3]):
        text = output.outputs[0].text if output.outputs else ""
        print(f"\n--- sample {i} ---\n{text[:2000]}")


def run_isolated(
    args: argparse.Namespace,
    dataset_name: str,
    mode: str,
    prompts: list[str],
) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue()
    process = ctx.Process(
        target=run_benchmark_child,
        args=(vars(args), dataset_name, mode, prompts, result_queue),
    )
    process.start()
    message = result_queue.get()
    process.join()

    if process.exitcode != 0:
        raise RuntimeError(
            f"Benchmark failed for dataset={dataset_name}, mode={mode}: {message}"
        )
    if not message["ok"]:
        raise RuntimeError(message)
    return message["result"]


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def print_summary(results: list[dict[str, Any]]) -> None:
    print("\nSummary")
    print(
        "dataset   mode      prompts  gen_s    gen_tok/s  req/s   load_s  total_s"
    )
    print("-" * 78)
    for row in results:
        print(
            f"{row['dataset']:<9} {row['mode']:<8} "
            f"{row['num_prompts']:>7} "
            f"{row['generation_seconds']:>7.2f} "
            f"{row['generated_tokens_per_second']:>9.2f} "
            f"{row['requests_per_second']:>6.2f} "
            f"{row['load_seconds']:>7.2f} "
            f"{row['total_seconds']:>7.2f}"
        )

    print("\nSpeedup from MTP over no_mtp, based on generation wall time")
    for dataset in sorted({row["dataset"] for row in results}):
        by_mode = {row["mode"]: row for row in results if row["dataset"] == dataset}
        if "no_mtp" not in by_mode or "mtp" not in by_mode:
            continue
        base = by_mode["no_mtp"]["generation_seconds"]
        mtp = by_mode["mtp"]["generation_seconds"]
        speedup = base / mtp if mtp > 0 else float("inf")
        delta = base - mtp
        print(f"{dataset}: {speedup:.3f}x ({delta:.2f}s faster)")


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    print(f"[speed.py] argv: {' '.join(sys.argv)}", flush=True)
    print(f"[speed.py] parsed args: {json.dumps(vars(args), sort_keys=True)}", flush=True)

    all_results: list[dict[str, Any]] = []
    for dataset_name in args.datasets:
        prompts = load_prompts(dataset_name, args.limit, args.dataset_cache_dir)
        for mode in args.modes:
            print(f"\nRunning {dataset_name} with mode={mode} on {len(prompts)} prompts")
            result = run_isolated(args, dataset_name, mode, prompts)
            all_results.append(result)
            write_jsonl(args.output, all_results)
            print_summary([result])

    write_jsonl(args.output, all_results)
    print_summary(all_results)
    if len(all_results) > 1:
        gen_times = [row["generation_seconds"] for row in all_results]
        print(f"\nMedian generation time: {statistics.median(gen_times):.2f}s")
    print(f"\nWrote results to {args.output}")


if __name__ == "__main__":
    main()
