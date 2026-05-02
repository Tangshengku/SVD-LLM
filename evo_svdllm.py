#coding:utf8
import argparse
import copy
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from SVDLLM import (
    _OnPolicyCovCollector,
    _build_gradient_whitening_profile,
    profle_svdllm_low_resource,
)
from component.low_rank_linear import LowRankLinear, ZeroLinear
from utils.data_utils import (
    _allocate_mixture_counts,
    _is_mixture_dataset,
    _load_training_texts,
    _split_mixture_dataset,
    get_calib_train_data,
    get_loaders,
    get_prompt_loaders,
)
from utils.model_utils import find_layers, get_model_from_huggingface


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def parse_max_memory(spec: Optional[str]):
    if spec is None or spec.strip() == "":
        return None
    stripped = spec.strip()
    if stripped.startswith("{"):
        parsed = json.loads(stripped)
        return {int(key) if str(key).isdigit() else key: value for key, value in parsed.items()}
    max_memory = {}
    for item in stripped.split(","):
        if not item.strip():
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        if key.isdigit():
            key = int(key)
        max_memory[key] = value.strip()
    return max_memory


def is_sharded_model(model) -> bool:
    hf_device_map = getattr(model, "hf_device_map", None)
    return isinstance(hf_device_map, dict) and len(set(hf_device_map.values())) > 1


def summarize_device_map(model) -> str:
    hf_device_map = getattr(model, "hf_device_map", None)
    if not isinstance(hf_device_map, dict):
        return "none"
    counts = {}
    for device in hf_device_map.values():
        counts[str(device)] = counts.get(str(device), 0) + 1
    return ", ".join(f"{device}:{count}" for device, count in sorted(counts.items()))


def model_input_device(model, fallback: str = "cuda") -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return torch.device(fallback)


def to_model_input(batch: torch.Tensor, model, fallback: str = "cuda") -> torch.Tensor:
    return batch.to(model_input_device(model, fallback))


def sample_top_p(logits, temperature=1.0, top_p=1.0):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = logits / temperature
    if top_p >= 1.0:
        probs = torch.softmax(scaled, dim=-1)
        return torch.multinomial(probs, num_samples=1)
    sorted_logits, sorted_indices = torch.sort(scaled, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_mask = cumulative_probs > top_p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False
    filtered_sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
    filtered_probs = torch.softmax(filtered_sorted_logits, dim=-1)
    sampled_sorted = torch.multinomial(filtered_probs, num_samples=1)
    return torch.gather(sorted_indices, -1, sampled_sorted)


def sample_on_policy_sequences_no_cache(model, prompts, rollout_len, temperature, top_p=1.0, device="cuda"):
    generated = to_model_input(prompts, model, device)
    for _ in range(rollout_len):
        logits = model(generated, use_cache=False).logits[:, -1, :].float()
        next_token = sample_top_p(logits, temperature=temperature, top_p=top_p)
        generated = torch.cat((generated, next_token), dim=1)
    return generated


def sample_on_policy_sequences(model, prompts, rollout_len, temperature, top_p=1.0, use_kv_cache=True, device="cuda"):
    if not use_kv_cache:
        return sample_on_policy_sequences_no_cache(model, prompts, rollout_len, temperature, top_p=top_p, device=device)
    generated = to_model_input(prompts, model, device)
    try:
        outputs = model(generated, use_cache=True)
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :].float()
        for step_idx in range(rollout_len):
            next_token = sample_top_p(logits, temperature=temperature, top_p=top_p)
            generated = torch.cat((generated, next_token), dim=1)
            if step_idx == rollout_len - 1:
                break
            outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :].float()
        return generated
    except Exception as err:
        log(f"KV-cache rollout failed ({type(err).__name__}: {err}); falling back to no-cache rollout")
        return sample_on_policy_sequences_no_cache(model, prompts, rollout_len, temperature, top_p=top_p, device=device)


def forward_kd_logits(model, sequences, prompt_len, rollout_len, slice_lm_head=True):
    start = prompt_len - 1
    end = start + rollout_len
    if start < 0 or end > sequences.shape[1]:
        raise ValueError("Invalid KD slice for sequence length")
    if not slice_lm_head:
        return model(sequences, use_cache=False).logits[:, start:end, :]
    try:
        if hasattr(model, "model") and hasattr(model, "lm_head"):
            outputs = model.model(sequences, use_cache=False)
            hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
            return model.lm_head(hidden_states[:, start:end, :])
    except Exception as err:
        log(f"Sliced KD logits failed ({type(err).__name__}: {err}); falling back to full logits")
    return model(sequences, use_cache=False).logits[:, start:end, :]


def compute_reverse_kd_loss(student_logits, teacher_logits, kd_temperature):
    student_log_prob = torch.log_softmax(student_logits.float() / kd_temperature, dim=-1)
    teacher_log_prob = torch.log_softmax(teacher_logits.float() / kd_temperature, dim=-1)
    return (kd_temperature ** 2) * torch.nn.functional.kl_div(
        teacher_log_prob,
        student_log_prob,
        reduction="batchmean",
        log_target=True,
    )


@dataclass
class WeightSearchSpace:
    name: str
    group: str
    in_features: int
    out_features: int
    dense_params: int
    max_rank: int
    rank_levels: List[int]
    source_names: List[str]
    singular_values_sq_by_source: List[torch.Tensor]
    left_u_by_source: List[torch.Tensor]
    right_v_by_source: List[torch.Tensor]
    bias: Optional[torch.Tensor]
    dtype: torch.dtype
    device: torch.device
    boundary_window: int
    tail_pool: List[int]

    def cost(self, rank: int) -> int:
        return rank * (self.in_features + self.out_features)

    def topk_selection(self, rank: int) -> List[int]:
        return list(range(rank))

    def singular_values_sq(self, source_idx: int) -> torch.Tensor:
        return self.singular_values_sq_by_source[source_idx]

    def left_u(self, source_idx: int) -> torch.Tensor:
        return self.left_u_by_source[source_idx]

    def right_v(self, source_idx: int) -> torch.Tensor:
        return self.right_v_by_source[source_idx]


def get_transformer_layers(model_name: str, model) -> Tuple[str, Sequence[nn.Module]]:
    if "opt" in model_name:
        return "model.decoder.layers", model.model.decoder.layers
    return "model.layers", model.model.layers


def space_layer_idx(space: WeightSearchSpace) -> Optional[int]:
    parts = space.name.split(".")
    for marker in ("layers",):
        if marker in parts:
            pos = parts.index(marker)
            if pos + 1 < len(parts):
                try:
                    return int(parts[pos + 1])
                except ValueError:
                    return None
    return None


def space_local_name(space: WeightSearchSpace) -> str:
    parts = space.name.split(".")
    if "layers" in parts:
        pos = parts.index("layers")
        if pos + 2 < len(parts):
            return ".".join(parts[pos + 2 :])
    return space.name.rsplit(".", 1)[-1]


def classify_group(weight_name: str) -> str:
    if any(token in weight_name for token in ["q_proj", "k_proj", "v_proj", "o_proj", "out_proj"]):
        return "attn"
    return "mlp"


def set_submodule(root: nn.Module, path: str, module: nn.Module) -> None:
    if "." in path:
        parent_path, attr_name = path.rsplit(".", 1)
        parent = root.get_submodule(parent_path)
    else:
        parent = root
        attr_name = path
    setattr(parent, attr_name, module)


def capture_dense_modules(model, spaces: Sequence[WeightSearchSpace]) -> List[nn.Module]:
    return [model.get_submodule(space.name) for space in spaces]


def refresh_space_runtime_metadata(model, spaces: Sequence[WeightSearchSpace]) -> None:
    for space in spaces:
        module = model.get_submodule(space.name)
        if not isinstance(module, nn.Linear):
            raise TypeError(f"Cached search space {space.name} no longer points to an nn.Linear module")
        if module.in_features != space.in_features or module.out_features != space.out_features:
            raise ValueError(
                f"Cached shape mismatch for {space.name}: cache=({space.out_features}, {space.in_features}) "
                f"model=({module.out_features}, {module.in_features})"
            )
        space.dtype = module.weight.dtype
        space.device = module.weight.device
        space.bias = clone_bias(module)


def clone_genome(genome: Dict[str, List[List[int]]]) -> Dict[str, List[List[int]]]:
    return {
        "ranks": list(genome["ranks"]),
        "selected": [list(item) for item in genome["selected"]],
        "sources": list(genome["sources"]),
    }


def save_evo_init_cache(
    path: str,
    args,
    spaces: Sequence[WeightSearchSpace],
    parent: Dict[str, List[List[int]]],
    total_dense_params: int,
    budget: int,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "version": 1,
        "model": args.model,
        "ratio": args.ratio,
        "model_seq_len": args.model_seq_len,
        "rank_step": args.rank_step,
        "boundary_window": args.boundary_window,
        "tail_count": args.tail_count,
        "init_strategy": args.init_strategy,
        "init_parent_method": args.init_parent_method,
        "init_parent_gradient_mode": args.init_parent_gradient_mode,
        "init_parent_input_covariance_source": args.init_parent_input_covariance_source,
        "init_parent_warm_start_ratio": args.init_parent_warm_start_ratio,
        "init_parent_on_policy_layer_tail_ratio": args.init_parent_on_policy_layer_tail_ratio,
        "source_datasets": args.source_datasets,
        "total_dense_params": total_dense_params,
        "budget": budget,
        "spaces": list(spaces),
        "parent": clone_genome(parent),
    }
    torch.save(payload, path)


def load_evo_init_cache(path: str, model, args) -> Tuple[List[WeightSearchSpace], Dict[str, List[List[int]]]]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    spaces = payload["spaces"]
    parent = clone_genome(payload["parent"])
    if len(spaces) != len(parent["ranks"]):
        raise ValueError(
            f"Invalid evo init cache: spaces={len(spaces)} but parent ranks={len(parent['ranks'])}"
        )
    if "ratio" in payload and abs(float(payload["ratio"]) - float(args.ratio)) > 1e-12:
        raise ValueError(
            f"Cached ratio {payload['ratio']} does not match --ratio {args.ratio}; "
            "use a matching cache or rerun initialization."
        )
    if "model_seq_len" in payload and int(payload["model_seq_len"]) != int(args.model_seq_len):
        log(
            f"Warning: cached model_seq_len={payload['model_seq_len']} differs from "
            f"--model_seq_len={args.model_seq_len}; continuing because search batches define eval length."
        )
    refresh_space_runtime_metadata(model, spaces)
    return spaces, parent


@torch.no_grad()
def apply_dense_modules(model, spaces: Sequence[WeightSearchSpace], dense_modules: Sequence[nn.Module]) -> None:
    for space, dense_module in zip(spaces, dense_modules):
        set_submodule(model, space.name, dense_module)


def capture_current_modules(model, spaces: Sequence[WeightSearchSpace]) -> List[nn.Module]:
    return [model.get_submodule(space.name) for space in spaces]


def clone_bias(module: nn.Linear) -> Optional[torch.Tensor]:
    if module.bias is None:
        return None
    return module.bias.detach().cpu().clone()


def parse_source_datasets(spec: str) -> List[str]:
    spec = spec.strip()
    if spec.startswith("mix:"):
        return [spec]
    datasets = [item.strip() for item in spec.split(",") if item.strip()]
    if not datasets:
        raise ValueError("At least one source dataset must be provided.")
    return datasets


def build_source_profile_plan(source_datasets: Sequence[str]) -> List[Tuple[str, str]]:
    if len(source_datasets) == 1 and source_datasets[0].startswith("mix:"):
        return [("mixed", source_datasets[0])]
    plan = [(dataset_name, dataset_name) for dataset_name in source_datasets]
    if len(source_datasets) > 1:
        mixed_dataset = "mix:" + ",".join(source_datasets)
        plan.append(("mixed", mixed_dataset))
    return plan


@torch.no_grad()
def build_search_spaces(
    model_name: str,
    model,
    profiling_mats: Dict[str, Dict[int, Dict[str, torch.Tensor]]],
    rank_step: int,
    boundary_window: int,
    tail_count: int,
    device: str,
) -> List[WeightSearchSpace]:
    layer_root, layers = get_transformer_layers(model_name, model)
    spaces: List[WeightSearchSpace] = []
    source_names = list(profiling_mats.keys())
    log(
        f"Building search spaces from {len(layers)} transformer layers "
        f"(rank_step={rank_step}, boundary_window={boundary_window}, tail_count={tail_count}, "
        f"sources={source_names})"
    )

    for layer_idx in tqdm(range(len(layers)), desc="Precomputing whitened SVD"):
        layer = layers[layer_idx]
        subset = find_layers(layer)
        for local_name, module in subset.items():
            if not isinstance(module, nn.Linear):
                continue
            weight = module.weight.detach().float().to(device)
            rows, cols = weight.shape
            dense_params = rows * cols
            full_rank = min(rows, cols)
            max_rank = min(full_rank, dense_params // (rows + cols))
            if max_rank <= 0:
                continue

            singular_values_sq_by_source = []
            left_u_by_source = []
            right_v_by_source = []
            for source_name in source_names:
                scaling_diag_matrix = profiling_mats[source_name][layer_idx][local_name].to(device).float()
                try:
                    scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
                except Exception:
                    scaling_diag_matrix = scaling_diag_matrix + 1e-6 * torch.eye(
                        scaling_diag_matrix.shape[0], device=device, dtype=scaling_diag_matrix.dtype
                    )
                    scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)

                whitened_weight = torch.matmul(weight, scaling_diag_matrix)
                u, s, vt = torch.linalg.svd(whitened_weight, full_matrices=False)
                right_v = torch.matmul(vt, scaling_matrix_inv).cpu()
                singular_values_sq_by_source.append(s.cpu() ** 2)
                left_u_by_source.append(u.cpu())
                right_v_by_source.append(right_v)
                del scaling_diag_matrix, scaling_matrix_inv, whitened_weight, u, s, vt, right_v

            rank_levels = sorted(set([0] + list(range(rank_step, max_rank + 1, rank_step)) + [max_rank]))

            boundary_cap = min(full_rank, max_rank + boundary_window + 1)
            tail_start = max(boundary_cap, full_rank - tail_count)
            tail_pool = list(range(tail_start, full_rank))

            spaces.append(
                WeightSearchSpace(
                    name=f"{layer_root}.{layer_idx}.{local_name}",
                    group=classify_group(local_name),
                    in_features=cols,
                    out_features=rows,
                    dense_params=dense_params,
                    max_rank=max_rank,
                    rank_levels=rank_levels,
                    source_names=list(source_names),
                    singular_values_sq_by_source=singular_values_sq_by_source,
                    left_u_by_source=left_u_by_source,
                    right_v_by_source=right_v_by_source,
                    bias=clone_bias(module),
                    dtype=module.weight.dtype,
                    device=module.weight.device,
                    boundary_window=boundary_window,
                    tail_pool=tail_pool,
                )
            )

            del weight
            torch.cuda.empty_cache()
    log(f"Finished SVD precomputation for {len(spaces)} searchable linear weights")
    return spaces


def level_index(space: WeightSearchSpace, rank: int) -> int:
    return space.rank_levels.index(rank)


def previous_rank(space: WeightSearchSpace, rank: int) -> Optional[int]:
    idx = level_index(space, rank)
    if idx == 0:
        return None
    return space.rank_levels[idx - 1]


def next_rank(space: WeightSearchSpace, rank: int) -> Optional[int]:
    idx = level_index(space, rank)
    if idx == len(space.rank_levels) - 1:
        return None
    return space.rank_levels[idx + 1]


def normalize_selection(space: WeightSearchSpace, source_idx: int, rank: int, selected: Sequence[int]) -> List[int]:
    if rank == 0:
        return []
    legal = [idx for idx in sorted(set(selected)) if 0 <= idx < len(space.singular_values_sq(source_idx))]
    if len(legal) > rank:
        legal = legal[:rank]
    if len(legal) < rank:
        for idx in range(len(space.singular_values_sq(source_idx))):
            if idx not in legal:
                legal.append(idx)
            if len(legal) == rank:
                break
    return sorted(legal)


def total_cost(spaces: Sequence[WeightSearchSpace], ranks: Sequence[int]) -> int:
    return sum(space.cost(rank) for space, rank in zip(spaces, ranks))


def rank_delta_utility(space: WeightSearchSpace, source_idx: int, old_rank: int, new_rank: int) -> float:
    lo = min(old_rank, new_rank)
    hi = max(old_rank, new_rank)
    if hi <= lo:
        return 0.0
    utility = space.singular_values_sq(source_idx)[lo:hi].sum().item()
    delta_cost = abs(space.cost(hi) - space.cost(lo))
    if delta_cost == 0:
        return float("inf")
    return utility / delta_cost


def greedy_initialization(spaces: Sequence[WeightSearchSpace], budget: int) -> List[int]:
    ranks = [0 for _ in spaces]
    current_cost = 0
    steps = 0
    while True:
        best_idx = None
        best_score = -1.0
        best_next_rank = None
        for idx, space in enumerate(spaces):
            next_level = next_rank(space, ranks[idx])
            if next_level is None:
                continue
            new_cost = current_cost + space.cost(next_level) - space.cost(ranks[idx])
            if new_cost > budget:
                continue
            score = rank_delta_utility(space, 0, ranks[idx], next_level)
            if score > best_score:
                best_idx = idx
                best_score = score
                best_next_rank = next_level
        if best_idx is None:
            break
        current_cost += spaces[best_idx].cost(best_next_rank) - spaces[best_idx].cost(ranks[best_idx])
        ranks[best_idx] = best_next_rank
        steps += 1
    log(f"Greedy initialization finished after {steps} rank-allocation steps")
    return ranks


def layer_key(space: WeightSearchSpace) -> str:
    parts = space.name.split(".")
    if "layers" in parts:
        layer_pos = parts.index("layers")
        if layer_pos + 1 < len(parts):
            return ".".join(parts[: layer_pos + 2])
    return space.name.rsplit(".", 1)[0]


def uniform_initialization(spaces: Sequence[WeightSearchSpace], budget: int) -> List[int]:
    ranks = [0 for _ in spaces]
    current_cost = 0
    steps = 0
    layer_to_indices: Dict[str, List[int]] = {}
    for idx, space in enumerate(spaces):
        layer_to_indices.setdefault(layer_key(space), []).append(idx)
    layer_order = sorted(layer_to_indices)

    while True:
        progressed = False
        for layer in layer_order:
            candidate_indices = []
            for idx in layer_to_indices[layer]:
                space = spaces[idx]
                next_level = next_rank(space, ranks[idx])
                if next_level is None:
                    continue
                delta_cost = space.cost(next_level) - space.cost(ranks[idx])
                if current_cost + delta_cost > budget:
                    continue
                fill = next_level / max(space.max_rank, 1)
                candidate_indices.append((fill, delta_cost, idx, next_level))
            if not candidate_indices:
                continue
            _, delta_cost, best_idx, best_next_rank = min(candidate_indices, key=lambda item: (item[0], item[1], item[2]))
            current_cost += delta_cost
            ranks[best_idx] = best_next_rank
            steps += 1
            progressed = True
        if not progressed:
            break
    log(f"Uniform initialization finished after {steps} strict per-layer rank-allocation steps")
    return ranks


def initialize_ranks(
    spaces: Sequence[WeightSearchSpace], budget: int, strategy: str
) -> List[int]:
    if strategy == "greedy":
        return greedy_initialization(spaces, budget)
    if strategy == "uniform":
        return uniform_initialization(spaces, budget)
    raise ValueError(f"Unsupported init strategy: {strategy}")


def build_topk_genome(spaces: Sequence[WeightSearchSpace], ranks: Sequence[int]) -> Dict[str, List[List[int]]]:
    default_source_idx = 0
    if spaces and "mixed" in spaces[0].source_names:
        default_source_idx = spaces[0].source_names.index("mixed")
    return {
        "ranks": list(ranks),
        "selected": [space.topk_selection(rank) for space, rank in zip(spaces, ranks)],
        "sources": [default_source_idx for _ in spaces],
    }


def set_genome_source(genome: Dict[str, List[List[int]]], spaces: Sequence[WeightSearchSpace], source_name: str) -> None:
    for idx, space in enumerate(spaces):
        if source_name not in space.source_names:
            raise ValueError(f"Source {source_name} is not available for {space.name}")
        source_idx = space.source_names.index(source_name)
        genome["sources"][idx] = source_idx
        genome["selected"][idx] = normalize_selection(
            space,
            source_idx,
            genome["ranks"][idx],
            genome["selected"][idx],
        )


def boundary_candidates(space: WeightSearchSpace, rank: int) -> List[int]:
    if rank == 0:
        return []
    start = max(0, rank - space.boundary_window)
    end = min(space.left_u_by_source[0].shape[1], rank + space.boundary_window)
    return list(range(start, end))


def mutate_rank_transfer(
    genome: Dict[str, List[List[int]]],
    spaces: Sequence[WeightSearchSpace],
    mutation_granularity: str,
) -> bool:
    def mutate_within_indices(candidate_indices: Sequence[int]) -> bool:
        donors = [idx for idx in candidate_indices if previous_rank(spaces[idx], genome["ranks"][idx]) is not None]
        receivers = [idx for idx in candidate_indices if next_rank(spaces[idx], genome["ranks"][idx]) is not None]
        if not donors or not receivers:
            return False

        donor = random.choice(donors)
        receivers = [idx for idx in receivers if idx != donor]
        if not receivers:
            return False
        receiver = random.choice(receivers)

        donor_rank = previous_rank(spaces[donor], genome["ranks"][donor])
        receiver_rank = next_rank(spaces[receiver], genome["ranks"][receiver])
        genome["ranks"][donor] = donor_rank
        genome["ranks"][receiver] = receiver_rank
        genome["selected"][donor] = normalize_selection(
            spaces[donor], genome["sources"][donor], donor_rank, genome["selected"][donor]
        )
        genome["selected"][receiver] = normalize_selection(
            spaces[receiver], genome["sources"][receiver], receiver_rank, genome["selected"][receiver]
        )
        return True

    if mutation_granularity == "group":
        mutated = False
        for group_name in ("attn", "mlp"):
            candidate_indices = [idx for idx, space in enumerate(spaces) if space.group == group_name]
            mutated = mutate_within_indices(candidate_indices) or mutated
        return mutated

    return mutate_within_indices(list(range(len(spaces))))


def mutate_boundary_swap(genome: Dict[str, List[List[int]]], spaces: Sequence[WeightSearchSpace]) -> bool:
    eligible = []
    for idx, space in enumerate(spaces):
        rank = genome["ranks"][idx]
        if rank <= 0:
            continue
        window = boundary_candidates(space, rank)
        retained = set(genome["selected"][idx])
        retained_boundary = [item for item in window if item in retained]
        dropped_boundary = [item for item in window if item not in retained]
        if retained_boundary and dropped_boundary:
            eligible.append((idx, retained_boundary, dropped_boundary))
    if not eligible:
        return False

    idx, retained_boundary, dropped_boundary = random.choice(eligible)
    selected = set(genome["selected"][idx])
    selected.remove(random.choice(retained_boundary))
    selected.add(random.choice(dropped_boundary))
    genome["selected"][idx] = normalize_selection(
        spaces[idx], genome["sources"][idx], genome["ranks"][idx], selected
    )
    return True


def mutate_tail_promotion(genome: Dict[str, List[List[int]]], spaces: Sequence[WeightSearchSpace]) -> bool:
    eligible = []
    for idx, space in enumerate(spaces):
        rank = genome["ranks"][idx]
        if rank <= 0 or not space.tail_pool:
            continue
        retained = set(genome["selected"][idx])
        window = boundary_candidates(space, rank)
        retained_boundary = [item for item in window if item in retained]
        tail = [item for item in space.tail_pool if item not in retained]
        if retained_boundary and tail:
            eligible.append((idx, retained_boundary, tail))
    if not eligible:
        return False

    idx, retained_boundary, tail = random.choice(eligible)
    selected = set(genome["selected"][idx])
    selected.remove(random.choice(retained_boundary))
    selected.add(random.choice(tail))
    genome["selected"][idx] = normalize_selection(
        spaces[idx], genome["sources"][idx], genome["ranks"][idx], selected
    )
    return True


def mutate_mask_reset(genome: Dict[str, List[List[int]]], spaces: Sequence[WeightSearchSpace]) -> bool:
    eligible = [idx for idx, rank in enumerate(genome["ranks"]) if rank > 0]
    if not eligible:
        return False
    idx = random.choice(eligible)
    genome["selected"][idx] = spaces[idx].topk_selection(genome["ranks"][idx])
    return True


def mutate_source_choice(genome: Dict[str, List[List[int]]], spaces: Sequence[WeightSearchSpace]) -> bool:
    eligible = [idx for idx, space in enumerate(spaces) if len(space.source_names) > 1]
    if not eligible:
        return False
    idx = random.choice(eligible)
    current_source = genome["sources"][idx]
    candidates = [source_idx for source_idx in range(len(spaces[idx].source_names)) if source_idx != current_source]
    if not candidates:
        return False
    genome["sources"][idx] = random.choice(candidates)
    genome["selected"][idx] = normalize_selection(
        spaces[idx], genome["sources"][idx], genome["ranks"][idx], genome["selected"][idx]
    )
    return True


def repair_budget(genome: Dict[str, List[List[int]]], spaces: Sequence[WeightSearchSpace], budget: int) -> None:
    while total_cost(spaces, genome["ranks"]) > budget:
        best_idx = None
        best_score = float("inf")
        best_prev_rank = None
        for idx, space in enumerate(spaces):
            prev_level = previous_rank(space, genome["ranks"][idx])
            if prev_level is None:
                continue
            score = rank_delta_utility(space, genome["sources"][idx], prev_level, genome["ranks"][idx])
            if score < best_score:
                best_idx = idx
                best_score = score
                best_prev_rank = prev_level
        if best_idx is None:
            raise RuntimeError("Unable to repair the genome to satisfy the budget.")
        genome["ranks"][best_idx] = best_prev_rank
        genome["selected"][best_idx] = normalize_selection(
            spaces[best_idx], genome["sources"][best_idx], best_prev_rank, genome["selected"][best_idx]
        )


def mutate_offspring(
    parent: Dict[str, List[List[int]]],
    spaces: Sequence[WeightSearchSpace],
    mutation_granularity: str,
    max_mutations: int,
    budget: int,
) -> Dict[str, List[List[int]]]:
    offspring = {
        "ranks": list(parent["ranks"]),
        "selected": [list(item) for item in parent["selected"]],
        "sources": list(parent["sources"]),
    }
    mutation_fns = [
        lambda genome: mutate_rank_transfer(genome, spaces, mutation_granularity),
        lambda genome: mutate_boundary_swap(genome, spaces),
        # lambda genome: mutate_source_choice(genome, spaces),
        # lambda genome: mutate_tail_promotion(genome, spaces),
        # lambda genome: mutate_mask_reset(genome, spaces),
    ]
    for _ in range(random.randint(1, max_mutations)):
        random.choice(mutation_fns)(offspring)
    repair_budget(offspring, spaces, budget)
    return offspring


def build_factor_weights(space: WeightSearchSpace, source_idx: int, selected: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    if not selected:
        return (
            torch.zeros((space.out_features, 0), dtype=space.dtype),
            torch.zeros((0, space.in_features), dtype=space.dtype),
        )
    idx = torch.tensor(sorted(selected), dtype=torch.long)
    # singular_values_sq stores s^2, so s^{1/2} = (s^2)^{1/4}.
    # Balanced split: left = U * s^{1/2}, right = s^{1/2} * right_v,
    # so left @ right = U @ diag(s) @ right_v (correct W approximation).
    sigma_half = space.singular_values_sq(source_idx)[idx].pow(0.25).to(torch.float32)
    left = space.left_u(source_idx)[:, idx].to(torch.float32) * sigma_half.unsqueeze(0)
    right = sigma_half.unsqueeze(1) * space.right_v(source_idx)[idx, :].to(torch.float32)

    # Balance every rank-1 component before casting to low precision.
    # This preserves the product U @ V but reduces peak magnitude and helps avoid fp16 overflow.
    left_max = left.abs().max(dim=0).values   # [rank]
    right_max = right.abs().max(dim=1).values  # [rank]
    nonzero = (left_max > 0) & (right_max > 0)
    scale = torch.ones(left.shape[1], dtype=torch.float32)
    scale[nonzero] = (right_max[nonzero] / left_max[nonzero]).sqrt()
    left = left * scale.unsqueeze(0)
    right = right / scale.unsqueeze(1)

    # Safety clamp: prevent fp16 overflow (max ~65504) from ill-conditioned whitening matrices.
    if space.dtype in (torch.float16, torch.bfloat16):
        fp_max = torch.finfo(space.dtype).max * 0.9
        left = left.clamp(-fp_max, fp_max)
        right = right.clamp(-fp_max, fp_max)

    u_weight = left.to(space.dtype)
    v_weight = right.to(space.dtype)
    return u_weight.cpu(), v_weight.cpu()


def _make_module(space: WeightSearchSpace, source_idx: int, rank: int, selected: Sequence[int]) -> nn.Module:
    if rank == 0:
        module = ZeroLinear(
            space.in_features,
            space.out_features,
            bias=space.bias,
            dtype=space.dtype,
            device=space.device,
        )
    else:
        u_weight, v_weight = build_factor_weights(space, source_idx, selected)
        if not torch.isfinite(u_weight).all():
            raise RuntimeError(
                f"Non-finite U factor generated for {space.name} at rank {rank} from source {space.source_names[source_idx]}"
            )
        if not torch.isfinite(v_weight).all():
            raise RuntimeError(
                f"Non-finite V factor generated for {space.name} at rank {rank} from source {space.source_names[source_idx]}"
            )
        module = LowRankLinear(
            space.in_features,
            space.out_features,
            u_weight.to(device=space.device, dtype=space.dtype),
            v_weight.to(device=space.device, dtype=space.dtype),
            bias=None if space.bias is None else space.bias.to(device=space.device, dtype=space.dtype),
        )
    return module.to(device=space.device, dtype=space.dtype)


def _build_evo_hook_specs(spaces: Sequence[WeightSearchSpace]) -> List[Tuple[int, str, str, str]]:
    return [(idx, space.name, space.name, space.name) for idx, space in enumerate(spaces)]


def select_tail_layer_space_indices(spaces: Sequence[WeightSearchSpace], layer_tail_ratio: float) -> List[int]:
    if not (0.0 <= layer_tail_ratio <= 1.0):
        raise ValueError("layer_tail_ratio must be in [0, 1]")
    layer_indices = sorted({idx for space in spaces for idx in [space_layer_idx(space)] if idx is not None})
    if not layer_indices:
        return list(range(len(spaces))) if layer_tail_ratio > 0 else []
    selected_layer_count = math.ceil(len(layer_indices) * layer_tail_ratio)
    if selected_layer_count == 0:
        return []
    first_layer = layer_indices[-selected_layer_count]
    return [
        idx for idx, space in enumerate(spaces)
        if (space_layer_idx(space) is not None and space_layer_idx(space) >= first_layer)
    ]


@torch.no_grad()
def _append_gradient_source_to_spaces(
    spaces: Sequence[WeightSearchSpace],
    dense_modules: Sequence[nn.Module],
    gradient_profile: Dict[int, Dict[str, Dict[str, torch.Tensor]]],
    source_name: str,
    device: str,
) -> None:
    if not spaces:
        return
    if source_name in spaces[0].source_names:
        raise ValueError(f"Source {source_name} already exists in search spaces")
    log(f"Appending gradient-whitened source '{source_name}' to {len(spaces)} search spaces")
    for idx, (space, dense_module) in enumerate(tqdm(zip(spaces, dense_modules), total=len(spaces), desc="Adding gradient source")):
        weight = dense_module.weight.detach().float().to(device)
        factors = gradient_profile[idx][space.name]
        input_factor = factors["x"].float().to(device)
        grad_factor = factors["g"].float().to(device)
        transformed = grad_factor.transpose(0, 1).matmul(weight).matmul(input_factor)
        u, s, vt = torch.linalg.svd(transformed, full_matrices=False)
        left = torch.linalg.solve(grad_factor.transpose(0, 1), u)
        right = torch.linalg.solve(input_factor.transpose(0, 1), vt.transpose(0, 1)).transpose(0, 1)
        space.source_names.append(source_name)
        space.singular_values_sq_by_source.append((s.cpu() ** 2))
        space.left_u_by_source.append(left.cpu())
        space.right_v_by_source.append(right.cpu())
        del weight, input_factor, grad_factor, transformed, u, s, vt, left, right
        torch.cuda.empty_cache()


def _build_tail_offline_input_profile(
    spaces: Sequence[WeightSearchSpace],
    parent: Dict[str, List[List[int]]],
    profiling_mats: Dict[str, Dict[int, Dict[str, torch.Tensor]]],
) -> Dict[int, Dict[str, Dict[str, torch.Tensor]]]:
    profile = {}
    for local_idx, space in enumerate(spaces):
        source_name = space.source_names[parent["sources"][local_idx]]
        layer_idx = space_layer_idx(space)
        if layer_idx is None:
            raise ValueError(f"Cannot infer transformer layer index from {space.name}")
        local_name = space_local_name(space)
        profile.setdefault(local_idx, {})[space.name] = profiling_mats[source_name][layer_idx][local_name].float().cpu()
    return profile


def add_on_policy_gradient_parent_source(
    model,
    spaces: Sequence[WeightSearchSpace],
    parent: Dict[str, List[List[int]]],
    dense_modules: Sequence[nn.Module],
    profiling_mats: Dict[str, Dict[int, Dict[str, torch.Tensor]]],
    prompts: Sequence[torch.Tensor],
    device: str,
    rollout_len: int,
    temperature: float,
    eval_batch_size: int,
    lambda0: float,
    whitening_mode: str,
    use_kv_cache: bool,
    slice_kd_lm_head: bool,
    layer_tail_ratio: float,
    input_covariance_source: str,
    source_name: str = "on_policy_grad",
) -> str:
    if input_covariance_source not in {"on_policy", "offline"}:
        raise ValueError("input_covariance_source must be one of: on_policy, offline")
    if not prompts:
        raise ValueError("on-policy gradient parent initialization requires non-empty prompts")
    tail_indices = select_tail_layer_space_indices(spaces, layer_tail_ratio)
    if not tail_indices:
        log("No tail layers selected for on-policy gradient parent initialization; keeping standard parent")
        return source_name
    tail_spaces = [spaces[idx] for idx in tail_indices]
    tail_dense_modules = [dense_modules[idx] for idx in tail_indices]
    tail_parent = {
        "ranks": [parent["ranks"][idx] for idx in tail_indices],
        "selected": [list(parent["selected"][idx]) for idx in tail_indices],
        "sources": [parent["sources"][idx] for idx in tail_indices],
    }
    prompt_len = prompts[0].shape[1]
    hook_specs = _build_evo_hook_specs(tail_spaces)
    collector = _OnPolicyCovCollector(
        model,
        hook_specs,
        alpha_min=0.0,
        alpha_max=float("inf"),
        delta=1e-6,
        device=device,
    )
    prompt_chunks = _iter_minibatches(prompts, eval_batch_size)
    parent_modules = capture_current_modules(model, spaces)
    log(
        f"Collecting on-policy gradient source for initial parent | prompts={len(prompts)} | "
        f"batches={len(prompt_chunks)} | prompt_len={prompt_len} | rollout_len={rollout_len} | "
        f"whitening_mode={whitening_mode} | tail_layer_ratio={layer_tail_ratio:.4f} | "
        f"tail_weights={len(tail_spaces)}/{len(spaces)} | "
        f"input_covariance_source={input_covariance_source}"
    )
    try:
        for batch_idx, prompt_chunk in enumerate(tqdm(prompt_chunks, desc="collecting parent on-policy gradients")):
            model.zero_grad(set_to_none=True)
            with torch.no_grad():
                sequences = sample_on_policy_sequences(
                    model,
                    prompt_chunk,
                    rollout_len=rollout_len,
                    temperature=temperature,
                    top_p=1.0,
                    use_kv_cache=use_kv_cache,
                    device=device,
                )
            collector.enabled = True
            student_logits = forward_kd_logits(
                model,
                sequences,
                prompt_len,
                rollout_len,
                slice_lm_head=slice_kd_lm_head,
            )
            apply_dense_modules(model, spaces, dense_modules)
            with torch.no_grad():
                teacher_logits = forward_kd_logits(
                    model,
                    sequences,
                    prompt_len,
                    rollout_len,
                    slice_lm_head=slice_kd_lm_head,
                )
            apply_dense_modules(model, spaces, parent_modules)
            loss = compute_reverse_kd_loss(
                student_logits,
                teacher_logits,
                kd_temperature=1.0,
            )
            loss.backward()
            collector.consume_batch((prompt_len - 1, prompt_len - 1 + rollout_len))
            collector.enabled = False
            model.zero_grad(set_to_none=True)
            if batch_idx == 0 or batch_idx == len(prompt_chunks) - 1:
                log(
                    f"Collected parent gradient batch {batch_idx + 1}/{len(prompt_chunks)} | "
                    f"reverse_kl={loss.item():.6f} | sequence_len={sequences.shape[1]}"
                )
    finally:
        collector.enabled = False
        collector.close()
        apply_dense_modules(model, spaces, parent_modules)
    offline_profile = None
    if input_covariance_source == "offline" and whitening_mode == "both":
        offline_profile = _build_tail_offline_input_profile(tail_spaces, tail_parent, profiling_mats)
    gradient_profile = _build_gradient_whitening_profile(
        collector.stats,
        lambda0=lambda0,
        dev=device,
        whitening_mode=whitening_mode,
        offline_profile=offline_profile,
    )
    _append_gradient_source_to_spaces(tail_spaces, tail_dense_modules, gradient_profile, source_name, device)
    for idx in tail_indices:
        space = spaces[idx]
        source_idx = space.source_names.index(source_name)
        parent["sources"][idx] = source_idx
        parent["selected"][idx] = normalize_selection(
            space,
            source_idx,
            parent["ranks"][idx],
            parent["selected"][idx],
        )
    apply_dense_modules(model, spaces, dense_modules)
    log(
        f"Initial parent switched to on-policy gradient source '{source_name}' "
        f"for {len(tail_indices)}/{len(spaces)} tail weights"
    )
    return source_name

@torch.no_grad()
def apply_genome(model, spaces: Sequence[WeightSearchSpace], genome: Dict[str, List[List[int]]]) -> None:
    for idx, space in enumerate(spaces):
        source_idx = genome["sources"][idx]
        selected = genome["selected"][idx]
        rank = genome["ranks"][idx]
        if len(selected) != rank:
            selected = normalize_selection(space, source_idx, rank, selected)
            genome["selected"][idx] = selected
        set_submodule(model, space.name, _make_module(space, source_idx, rank, selected))


@torch.no_grad()
def apply_genome_diff(
    model,
    spaces: Sequence[WeightSearchSpace],
    base_genome: Dict[str, List[List[int]]],
    new_genome: Dict[str, List[List[int]]],
) -> List[Tuple[str, nn.Module]]:
    """Apply only the spaces that changed relative to base_genome.

    Returns a rollback list of (module_path, old_module) so the base genome
    can be restored cheaply with rollback_modules().
    """
    rollback: List[Tuple[str, nn.Module]] = []
    for idx, space in enumerate(spaces):
        if (
            base_genome["ranks"][idx] == new_genome["ranks"][idx]
            and base_genome["selected"][idx] == new_genome["selected"][idx]
            and base_genome["sources"][idx] == new_genome["sources"][idx]
        ):
            continue
        rollback.append((space.name, model.get_submodule(space.name)))
        source_idx = new_genome["sources"][idx]
        selected = new_genome["selected"][idx]
        rank = new_genome["ranks"][idx]
        if len(selected) != rank:
            selected = normalize_selection(space, source_idx, rank, selected)
            new_genome["selected"][idx] = selected
        set_submodule(model, space.name, _make_module(space, source_idx, rank, selected))
    return rollback


def rollback_modules(model: nn.Module, rollback: List[Tuple[str, nn.Module]]) -> None:
    for name, module in rollback:
        set_submodule(model, name, module)


def _iter_minibatches(
    samples: Sequence[torch.Tensor], batch_size: int
) -> List[torch.Tensor]:
    """Concatenate samples into chunks of batch_size along dim 0."""
    chunks = []
    for start in range(0, len(samples), batch_size):
        chunks.append(torch.cat(list(samples[start : start + batch_size]), dim=0))
    return chunks


@torch.no_grad()
def compute_nll(model, batches: Sequence[torch.Tensor], device: str, eval_batch_size: int = 1) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for chunk in tqdm(_iter_minibatches(batches, eval_batch_size), desc="computing nll"):
        chunk = to_model_input(chunk, model, device)       # [bs, seqlen]
        logits = model(chunk, use_cache=False).logits.float()  # fp32 prevents log_softmax underflow
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = chunk[:, 1:].to(shift_logits.device).contiguous()
        n_tokens = shift_labels.numel()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += n_tokens
    return total_loss / total_tokens


@torch.no_grad()
def precompute_teacher_logits(model, batches: Sequence[torch.Tensor], device: str) -> List[torch.Tensor]:
    targets = []
    model.eval()
    log(f"Starting dense teacher-logit precomputation on {len(batches)} search batches")
    for batch_idx, batch in enumerate(tqdm(batches, desc="Computing dense teacher logits")):
        logits = model(to_model_input(batch, model, device), use_cache=False).logits[:, :-1, :].float()
        if not torch.isfinite(logits).all():
            raise RuntimeError(f"Non-finite dense teacher logits detected on batch {batch_idx}")
        logits = logits.cpu()
        targets.append(logits)
    log("Finished dense teacher-logit precomputation")
    return targets


@torch.no_grad()
def compute_kl(
    model,
    batches: Sequence[torch.Tensor],
    teacher_logits: Optional[Sequence[torch.Tensor]],
    device: str,
    eval_batch_size: int = 1,
    spaces: Optional[Sequence[WeightSearchSpace]] = None,
    dense_modules: Optional[Sequence[nn.Module]] = None,
) -> float:
    model.eval()
    if teacher_logits is None and (spaces is None or dense_modules is None):
        raise ValueError("Streaming KL requires spaces and dense_modules when teacher_logits is None")
    total_loss = 0.0
    total_seqs = 0
    batch_chunks = _iter_minibatches(batches, eval_batch_size)
    teacher_chunks = _iter_minibatches(teacher_logits, eval_batch_size) if teacher_logits is not None else [None] * len(batch_chunks)
    candidate_modules = capture_current_modules(model, spaces) if teacher_logits is None else None
    for chunk, teacher_chunk in tqdm(zip(batch_chunks, teacher_chunks), desc="computing kl"):
        chunk = to_model_input(chunk, model, device)
        student_logits = model(chunk, use_cache=False).logits[:, :-1, :].float()
        if not torch.isfinite(student_logits).all():
            raise RuntimeError("Non-finite student logits detected during KL computation")
        if teacher_chunk is None:
            apply_dense_modules(model, spaces, dense_modules)
            try:
                teacher_chunk = model(chunk, use_cache=False).logits[:, :-1, :].float()
            finally:
                apply_dense_modules(model, spaces, candidate_modules)
        else:
            teacher_chunk = teacher_chunk.to(student_logits.device)
        teacher_log_prob = torch.log_softmax(teacher_chunk.to(student_logits.device), dim=-1)
        student_log_prob = torch.log_softmax(student_logits, dim=-1)
        # batchmean divides by batch size; accumulate as sum then divide once at end
        loss = torch.nn.functional.kl_div(
            student_log_prob,
            teacher_log_prob,
            reduction="sum",
            log_target=True,
        )
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite KL loss detected")
        total_loss += loss.item()
        total_seqs += chunk.shape[0]
    # Normalize by total number of (sequence, position) pairs to match batchmean semantics
    seqlen_minus_one = batches[0].shape[1] - 1
    return total_loss / (total_seqs * seqlen_minus_one)


@torch.no_grad()
def compute_on_policy_kl(
    model,
    spaces: Sequence[WeightSearchSpace],
    genome: Dict[str, List[List[int]]],
    dense_modules: Sequence[nn.Module],
    prompts: Sequence[torch.Tensor],
    device: str,
    rollout_len: int,
    eval_batch_size: int = 1,
    temperature: float = 1.0,
    eval_every: int = 1,
) -> float:
    if rollout_len <= 0:
        raise ValueError("on-policy rollout length must be positive")
    if not prompts:
        raise ValueError("on-policy prompts must not be empty")
    if eval_every <= 0:
        raise ValueError("on-policy evaluation stride must be positive")

    model.eval()
    total_loss = 0.0
    total_steps = 0
    prompt_chunks = _iter_minibatches(prompts, eval_batch_size)
    candidate_modules = capture_current_modules(model, spaces)

    for prompt_chunk in tqdm(prompt_chunks, desc="computing on-policy kl"):
        generated = to_model_input(prompt_chunk, model, device)
        rollout_contexts = []
        student_logits_per_step = []

        for step_idx in range(rollout_len):
            should_eval = ((step_idx + 1) % eval_every == 0) or (step_idx == rollout_len - 1)
            if should_eval:
                rollout_contexts.append(generated.cpu())
            next_logits = model(generated, use_cache=False).logits[:, -1, :].float()
            if not torch.isfinite(next_logits).all():
                raise RuntimeError("Non-finite student logits detected during on-policy KL rollout")
            if should_eval:
                student_logits_per_step.append(next_logits.cpu())
            sample_probs = torch.softmax(next_logits / temperature, dim=-1)
            next_token = torch.multinomial(sample_probs, num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
        apply_dense_modules(model, spaces, dense_modules)
        try:
            for context, student_logits in zip(rollout_contexts, student_logits_per_step):
                teacher_logits = model(to_model_input(context, model, device), use_cache=False).logits[:, -1, :].float()
                teacher_log_prob = torch.log_softmax(teacher_logits, dim=-1)
                student_log_prob = torch.log_softmax(student_logits.to(teacher_logits.device), dim=-1)
                loss = torch.nn.functional.kl_div(
                    teacher_log_prob,
                    student_log_prob,
                    reduction="sum",
                    log_target=True,
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite reverse on-policy KL loss detected")
                total_loss += loss.item()
                total_steps += context.shape[0]
        finally:
            apply_dense_modules(model, spaces, candidate_modules)

    return total_loss / total_steps


@torch.no_grad()
def compute_fitness(
    model,
    batches: Sequence[torch.Tensor],
    device: str,
    fitness_fn: str,
    teacher_logits: Optional[Sequence[torch.Tensor]] = None,
    alpha: float = 0.5,
    eval_batch_size: int = 1,
    spaces: Optional[Sequence[WeightSearchSpace]] = None,
    genome: Optional[Dict[str, List[List[int]]]] = None,
    dense_modules: Optional[Sequence[nn.Module]] = None,
    on_policy_rollout_len: int = 32,
    on_policy_temperature: float = 1.0,
    on_policy_eval_every: int = 1,
) -> float:
    """Evaluate fitness of the model in its current state (genome already applied)."""
    if fitness_fn == "ppl":
        return math.exp(compute_nll(model, batches, device, eval_batch_size))
    if fitness_fn == "kl":
        return compute_kl(
            model,
            batches,
            teacher_logits,
            device,
            eval_batch_size,
            spaces=spaces,
            dense_modules=dense_modules,
        )
    if fitness_fn == "on_policy_kl":
        if spaces is None or genome is None or dense_modules is None:
            raise ValueError("on_policy_kl requires spaces, genome, and dense_modules")
        return compute_on_policy_kl(
            model,
            spaces,
            genome,
            dense_modules,
            batches,
            device,
            rollout_len=on_policy_rollout_len,
            eval_batch_size=eval_batch_size,
            temperature=on_policy_temperature,
            eval_every=on_policy_eval_every,
        )
    nll = compute_nll(model, batches, device, eval_batch_size)
    kl = compute_kl(
        model,
        batches,
        teacher_logits,
        device,
        eval_batch_size,
        spaces=spaces,
        dense_modules=dense_modules,
    )
    return alpha * kl + (1.0 - alpha) * nll


@torch.no_grad()
def evaluate_genome(
    model,
    spaces: Sequence[WeightSearchSpace],
    genome: Dict[str, List[List[int]]],
    batches: Sequence[torch.Tensor],
    device: str,
    fitness_fn: str,
    teacher_logits: Optional[Sequence[torch.Tensor]] = None,
    alpha: float = 0.5,
    eval_batch_size: int = 1,
    dense_modules: Optional[Sequence[nn.Module]] = None,
    on_policy_rollout_len: int = 32,
    on_policy_temperature: float = 1.0,
    on_policy_eval_every: int = 1,
) -> float:
    apply_genome(model, spaces, genome)
    return compute_fitness(
        model,
        batches,
        device,
        fitness_fn,
        teacher_logits,
        alpha,
        eval_batch_size,
        spaces=spaces,
        genome=genome,
        dense_modules=dense_modules,
        on_policy_rollout_len=on_policy_rollout_len,
        on_policy_temperature=on_policy_temperature,
        on_policy_eval_every=on_policy_eval_every,
    )


def get_search_batches(dataset: str, tokenizer, nsamples: int, seqlen: int, seed: int) -> List[torch.Tensor]:
    if dataset.startswith("mix:") or "evol-codealpaca" in dataset.lower() or "tulu" in dataset.lower():
        log(
            f"Sampling search batches from streaming text chunks | dataset={dataset} | "
            f"nsamples={nsamples} | seqlen={seqlen}"
        )
        return get_streaming_text_batches(dataset, tokenizer, nsamples, seqlen, seed)
    loader, _ = get_loaders(dataset, nsamples=nsamples, seed=seed, tokenizer=tokenizer, seqlen=seqlen)
    return [inp for inp, _ in loader]


def get_streaming_text_batches(dataset: str, tokenizer, nsamples: int, seqlen: int, seed: int) -> List[torch.Tensor]:
    if _is_mixture_dataset(dataset):
        dataset_names = _split_mixture_dataset(dataset)
        counts = _allocate_mixture_counts(nsamples, len(dataset_names))
        batches = []
        for idx, (dataset_name, count) in enumerate(zip(dataset_names, counts)):
            if count <= 0:
                continue
            log(
                f"Loading search source {idx + 1}/{len(dataset_names)} | "
                f"dataset={dataset_name} | nsamples={count}"
            )
            batches.extend(
                _sample_streaming_text_batches(
                    _load_training_texts(dataset_name),
                    tokenizer,
                    count,
                    seqlen,
                    seed + idx,
                )
            )
            log(
                f"Search source ready {idx + 1}/{len(dataset_names)} | "
                f"dataset={dataset_name} | total_batches={len(batches)}"
            )
        return batches
    return _sample_streaming_text_batches(
        _load_training_texts(dataset),
        tokenizer,
        nsamples,
        seqlen,
        seed,
    )


def _sample_streaming_text_batches(texts: Sequence[str], tokenizer, nsamples: int, seqlen: int, seed: int) -> List[torch.Tensor]:
    rng = random.Random(seed)
    batches = []
    max_attempts = max(nsamples * 50, 200)
    for _ in range(max_attempts):
        if len(batches) >= nsamples:
            break
        start = rng.randint(0, len(texts) - 1)
        parts = []
        total_chars = 0
        for offset in range(min(len(texts), 128)):
            text = texts[(start + offset) % len(texts)]
            if text:
                parts.append(text)
                total_chars += len(text)
            if total_chars >= seqlen * 8:
                break
        if not parts:
            continue
        enc = tokenizer("\n\n".join(parts), return_tensors="pt")
        if enc.input_ids.shape[1] < seqlen:
            continue
        token_start = rng.randint(0, enc.input_ids.shape[1] - seqlen)
        batches.append(enc.input_ids[:, token_start : token_start + seqlen].contiguous())
    if len(batches) < nsamples:
        raise ValueError(
            f"Only built {len(batches)}/{nsamples} search batches with seqlen={seqlen}; "
            "try lowering --model_seq_len or using more/longer text."
        )
    return batches


def build_on_policy_prompts(batches: Sequence[torch.Tensor], prompt_len: int) -> List[torch.Tensor]:
    if prompt_len <= 0:
        raise ValueError("on-policy prompt length must be positive")
    prompts = []
    for batch in batches:
        if batch.shape[1] <= prompt_len:
            raise ValueError(
                f"on-policy prompt length {prompt_len} must be smaller than the search batch sequence length {batch.shape[1]}"
            )
        prompts.append(batch[:, :prompt_len].contiguous())
    return prompts


def fitness_requires_teacher_logits(fitness_fn: str) -> bool:
    return fitness_fn in {"kl", "hyb"}


def genome_to_serializable(spaces: Sequence[WeightSearchSpace], genome: Dict[str, List[List[int]]]) -> Dict[str, object]:
    per_weight = []
    for idx, (space, rank, selected) in enumerate(zip(spaces, genome["ranks"], genome["selected"])):
        per_weight.append(
            {
                "name": space.name,
                "group": space.group,
                "rank": rank,
                "cost": space.cost(rank),
                "selected": list(selected),
                "source": space.source_names[genome["sources"][idx]],
            }
        )
    return {
        "total_cost": total_cost(spaces, genome["ranks"]),
        "source_datasets": list(spaces[0].source_names) if spaces else [],
        "weights": per_weight,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Dense Hugging Face model name or local path.")
    parser.add_argument("--ratio", type=float, default=0.2, help="Target compression ratio in parameter reduction.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext2",
        help="Calibration/search dataset. Supports single datasets and mixtures like mix:wikitext2,evol-codealpaca,tulu-math.",
    )
    parser.add_argument(
        "--source_datasets",
        type=str,
        default="wikitext2,evol-codealpaca,tulu-math",
        help="Comma-separated datasets used to build source-specific whitening profiles. When multiple datasets are given, an additional mixed source is built automatically and used for parent initialization. Passing a single mix:... spec uses only the mixed source and disables source mutation.",
    )
    parser.add_argument("--whitening_nsamples", type=int, default=256, help="Calibration samples for whitening.")
    parser.add_argument(
        "--profile_batch_size",
        type=int,
        default=8,
        help="Mini-batch size used inside low-resource whitening profiling.",
    )
    parser.add_argument("--search_nsamples", type=int, default=16, help="Calibration samples for evolutionary search.")
    parser.add_argument("--model_seq_len", type=int, default=2048, help="Sequence length.")
    parser.add_argument("--profiling_mat_path", type=str, default=None, help="Load precomputed whitening matrices.")
    parser.add_argument(
        "--evo_init_cache_path",
        type=str,
        default=None,
        help=(
            "Load cached evolutionary initialization state containing precomputed search spaces, "
            "including any on-policy gradient source, and the initial parent genome."
        ),
    )
    parser.add_argument(
        "--save_evo_init_cache_path",
        type=str,
        default=None,
        help=(
            "Save evolutionary initialization state after the initial parent is prepared. "
            "This avoids recomputing offline SVD and on-policy gradient whitening on later runs."
        ),
    )
    parser.add_argument("--save_path", type=str, default=None, help="Directory for configs or model checkpoints.")
    parser.add_argument("--save_model", action="store_true", help="Save the final compressed model checkpoint.")
    parser.add_argument(
        "--fitness_fn",
        choices=["ppl", "kl", "hyb", "on_policy_kl"],
        default="kl",
        help="Search fitness.",
    )
    parser.add_argument("--hybrid_alpha", type=float, default=0.5, help="Weight of KL in hybrid fitness.")
    parser.add_argument(
        "--on_policy_prompt_len",
        type=int,
        default=128,
        help="Prompt length used to start on-policy KL rollouts.",
    )
    parser.add_argument(
        "--on_policy_rollout_len",
        type=int,
        default=128,
        help="Rollout length used for on-policy KL fitness.",
    )
    parser.add_argument(
        "--on_policy_temperature",
        type=float,
        default=1.0,
        help="Sampling temperature used for on-policy KL rollouts.",
    )
    parser.add_argument(
        "--on_policy_eval_every",
        type=int,
        default=1,
        help="Evaluate teacher KL only every k rollout steps during on-policy KL.",
    )
    parser.add_argument(
        "--rerank_topk_on_policy",
        type=int,
        default=0,
        help="If > 0, score all candidates with --rerank_base_fitness and rerank only the top-k candidates with on-policy KL.",
    )
    parser.add_argument(
        "--rerank_base_fitness",
        choices=["ppl", "kl", "hyb"],
        default="kl",
        help="Base fitness used before top-k on-policy KL reranking.",
    )
    parser.add_argument(
        "--rerank_selection_fitness",
        choices=["on_policy_kl", "on_policy_plus_kl"],
        default="on_policy_kl",
        help="Final selection metric on the top-k shortlist: either pure on-policy KL or on-policy KL plus teacher-forced KL.",
    )
    parser.add_argument(
        "--rerank_on_policy_weight",
        type=float,
        default=1.0,
        help="Scalar weight applied to on-policy KL inside --rerank_selection_fitness=on_policy_plus_kl.",
    )
    parser.add_argument("--generations", type=int, default=50, help="Number of search generations.")
    parser.add_argument("--offspring", type=int, default=8, help="Number of offspring per generation.")
    parser.add_argument("--max_mutations", type=int, default=3, help="Maximum mutations per offspring.")
    parser.add_argument("--rank_step", type=int, default=8, help="Rank step for admissible levels.")
    parser.add_argument("--boundary_window", type=int, default=8, help="Boundary search window size.")
    parser.add_argument("--tail_count", type=int, default=8, help="Tail-pool size per weight.")
    parser.add_argument("--mutation_granularity", choices=["group", "weight"], default="group")
    parser.add_argument(
        "--init_strategy",
        choices=["greedy", "uniform"],
        default="greedy",
        help="Initial rank-allocation strategy for the first genome.",
    )
    parser.add_argument(
        "--init_parent_method",
        choices=["standard", "on_policy_gradient"],
        default="standard",
        help="How to choose the initial parent source. on_policy_gradient collects step-6-style gradient whitening stats from the standard warm-start parent and uses that as the initial source.",
    )
    parser.add_argument(
        "--init_parent_gradient_mode",
        choices=["both", "grad_only"],
        default="both",
        help="Gradient-whitening mode used by --init_parent_method on_policy_gradient.",
    )
    parser.add_argument(
        "--init_parent_input_covariance_source",
        choices=["on_policy", "offline"],
        default="on_policy",
        help="Input covariance source for --init_parent_method on_policy_gradient when --init_parent_gradient_mode both is used.",
    )
    parser.add_argument(
        "--init_parent_warm_start_ratio",
        type=float,
        default=None,
        help="Optional compression ratio for the warm-start student used to collect on-policy gradients. Uses the same convention as --ratio. If omitted, uses --ratio.",
    )
    parser.add_argument(
        "--init_parent_lambda0",
        type=float,
        default=1e-6,
        help="Diagonal stabilization scale for on-policy gradient parent initialization.",
    )
    parser.add_argument(
        "--init_parent_on_policy_layer_tail_ratio",
        type=float,
        default=1.0,
        help="Fraction of final transformer layers that receive the on_policy_grad source during on-policy gradient parent initialization. Earlier layers keep the original offline source.",
    )
    parser.add_argument(
        "--init_parent_on_policy_nsamples",
        type=int,
        default=None,
        help="Number of prompt samples used only for on-policy gradient parent initialization. If omitted, uses --search_nsamples.",
    )
    parser.add_argument(
        "--init_parent_on_policy_dataset",
        type=str,
        default=None,
        help="Dataset used only for on-policy gradient parent initialization prompts. If omitted, uses --dataset.",
    )
    parser.add_argument(
        "--disable_init_parent_kv_cache",
        action="store_true",
        help="Disable KV-cache rollout generation during on-policy gradient parent initialization.",
    )
    parser.add_argument(
        "--disable_init_parent_lm_head_slicing",
        action="store_true",
        help="Disable sliced lm_head logits during on-policy gradient parent initialization.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--DEV", type=str, default="cuda", help="Search device.")
    parser.add_argument(
        "--model_device_map",
        type=str,
        default="auto",
        help=(
            "Transformers device_map used for dense/search model loading. "
            "Use 'auto' to shard large models across GPUs with accelerate, or 'none' for the old explicit model.to(--DEV) path."
        ),
    )
    parser.add_argument(
        "--model_max_memory",
        type=str,
        default=None,
        help=(
            "Optional max_memory for Transformers device_map, either JSON or comma form like "
            "'0=78GiB,1=78GiB,cpu=200GiB'."
        ),
    )
    parser.add_argument(
        "--no_flash_attention_2",
        action="store_true",
        help="Disable flash_attention_2 when loading the model.",
    )
    parser.add_argument(
        "--teacher_logits_mode",
        choices=["cache", "stream"],
        default="cache",
        help=(
            "For KL fitness, cache dense teacher logits once, or stream/recompute them per evaluation minibatch. "
            "Use stream for large-vocab large-model runs to avoid huge CPU RAM use."
        ),
    )
    parser.add_argument("--eval_batch_size", type=int, default=4, help="Mini-batch size for fitness forward passes.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.rerank_topk_on_policy < 0:
        raise ValueError("--rerank_topk_on_policy must be non-negative")
    if args.rerank_on_policy_weight < 0:
        raise ValueError("--rerank_on_policy_weight must be non-negative")
    if args.init_parent_warm_start_ratio is not None and not (0.0 <= args.init_parent_warm_start_ratio < 1.0):
        raise ValueError("--init_parent_warm_start_ratio must be in [0, 1)")
    if not (0.0 <= args.init_parent_on_policy_layer_tail_ratio <= 1.0):
        raise ValueError("--init_parent_on_policy_layer_tail_ratio must be in [0, 1]")
    if args.init_parent_on_policy_nsamples is not None and args.init_parent_on_policy_nsamples <= 0:
        raise ValueError("--init_parent_on_policy_nsamples must be positive")
    if args.fitness_fn == "on_policy_kl" and args.rerank_topk_on_policy > 0:
        log(
            "Direct on-policy KL mode selected via --fitness_fn on_policy_kl; "
            "disabling top-k rerank and scoring all candidates with on-policy KL."
        )
        args.rerank_topk_on_policy = 0
    source_datasets = parse_source_datasets(args.source_datasets)
    source_profile_plan = build_source_profile_plan(source_datasets)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model_device_map = None if args.model_device_map.lower() == "none" else args.model_device_map
    model_max_memory = parse_max_memory(args.model_max_memory)
    attn_implementation = None if args.no_flash_attention_2 else "flash_attention_2"
    run_start = time.time()
    rerank_enabled = args.rerank_topk_on_policy > 0
    bulk_fitness_fn = args.rerank_base_fitness if rerank_enabled else args.fitness_fn
    selection_fitness_name = args.rerank_selection_fitness if rerank_enabled else args.fitness_fn

    log(
        f"Launching evolutionary SVD search | model={args.model} | ratio={args.ratio} | "
        f"dataset={args.dataset} | fitness={args.fitness_fn} | generations={args.generations} | "
        f"offspring={args.offspring} | sources={source_datasets} | "
        f"profile_plan={[name for name, _ in source_profile_plan]} | device={args.DEV} | "
        f"model_device_map={model_device_map if model_device_map is not None else 'none'}"
    )

    log("Loading dense model and tokenizer")
    model, tokenizer = get_model_from_huggingface(
        args.model,
        device_map=model_device_map,
        max_memory=model_max_memory,
        attn_implementation=attn_implementation,
    )
    model.eval()
    model.seqlen = args.model_seq_len
    if model_device_map is None:
        model = model.to(args.DEV)
    elif is_sharded_model(model):
        log(f"Loaded sharded model with device placement summary: {summarize_device_map(model)}")
    model.config.use_cache = False
    log(f"Loaded model; sequence length set to {args.model_seq_len}")

    loaded_evo_init_cache = args.evo_init_cache_path is not None
    profiling_mats = None
    if loaded_evo_init_cache:
        log(f"Loading cached evolutionary initialization from {args.evo_init_cache_path}")
        spaces, parent = load_evo_init_cache(args.evo_init_cache_path, model, args)
        if not spaces:
            raise RuntimeError("Cached evolutionary initialization contains no search spaces.")
        dense_modules = capture_dense_modules(model, spaces)
        total_dense_params = sum(space.dense_params for space in spaces)
        budget = int((1.0 - args.ratio) * total_dense_params)
        log(
            f"Loaded cached search state | weights={len(spaces)} | "
            f"dense_params={total_dense_params} | target_kept_budget={budget} | "
            f"kept_params={total_cost(spaces, parent['ranks'])} | "
            f"default_source={spaces[0].source_names[parent['sources'][0]] if spaces else 'n/a'}"
        )
    else:
        if args.profiling_mat_path is None:
            profiling_mats = {}
            for source_idx, (source_name, calibration_dataset) in enumerate(source_profile_plan):
                log(
                    f"Collecting whitening stats for source={source_name} "
                    f"from dataset={calibration_dataset} with {args.whitening_nsamples} calibration samples"
                )
                whitening_data = get_calib_train_data(
                    calibration_dataset,
                    tokenizer,
                    args.whitening_nsamples,
                    seqlen=args.model_seq_len,
                    seed=args.seed + source_idx,
                )
                profiling_mats[source_name] = profle_svdllm_low_resource(
                    args.model, model, whitening_data, args.DEV, profile_batch_size=args.profile_batch_size
                )
            log("Whitening/profile collection finished")
        else:
            if len(source_profile_plan) != 1:
                raise ValueError(
                    "--profiling_mat_path currently supports only a single source dataset. "
                    "Leave it unset to compute multiple source-specific profiles."
                )
            log(f"Loading profiling matrices from {args.profiling_mat_path}")
            profiling_mats = {source_profile_plan[0][0]: torch.load(args.profiling_mat_path, map_location="cpu")}
            log("Loaded profiling matrices from disk")

        # The low-resource profiling path moves major submodules back to CPU. For
        # device_map loading, reload to restore the accelerate shard placement.
        if model_device_map is None:
            model = model.to(args.DEV)
            log(f"Dense model moved back to {args.DEV} for search-time evaluation")
        elif args.profiling_mat_path is None:
            log("Reloading dense model with device_map for sharded search-time evaluation")
            del model
            torch.cuda.empty_cache()
            model, _ = get_model_from_huggingface(
                args.model,
                device_map=model_device_map,
                max_memory=model_max_memory,
                attn_implementation=attn_implementation,
            )
            model.seqlen = args.model_seq_len
            if is_sharded_model(model):
                log(f"Reloaded sharded model with device placement summary: {summarize_device_map(model)}")
        model.eval()
        model.config.use_cache = False

        spaces = build_search_spaces(
            args.model,
            model,
            profiling_mats,
            rank_step=args.rank_step,
            boundary_window=args.boundary_window,
            tail_count=args.tail_count,
            device=args.DEV,
        )
        if not spaces:
            raise RuntimeError("No admissible linear weights found for evolutionary SVD search.")
        dense_modules = capture_dense_modules(model, spaces)

        total_dense_params = sum(space.dense_params for space in spaces)
        budget = int((1.0 - args.ratio) * total_dense_params)
        attn_weights = sum(1 for space in spaces if space.group == "attn")
        mlp_weights = sum(1 for space in spaces if space.group == "mlp")
        log(
            f"Search space ready: {len(spaces)} weights "
            f"({attn_weights} attention, {mlp_weights} mlp) | "
            f"dense_params={total_dense_params} | target_kept_budget={budget}"
        )
        initial_ranks = initialize_ranks(spaces, budget, args.init_strategy)
        parent = build_topk_genome(spaces, initial_ranks)
        repair_budget(parent, spaces, budget)
        log(
            f"Initial genome prepared | init_strategy={args.init_strategy} | "
            f"kept_params={total_cost(spaces, parent['ranks'])} | "
            f"active_weights={sum(rank > 0 for rank in parent['ranks'])} | "
            f"default_source={spaces[0].source_names[parent['sources'][0]] if spaces else 'n/a'}"
        )

    log(f"Preparing {args.search_nsamples} search batches from {args.dataset}")
    search_batches = get_search_batches(args.dataset, tokenizer, args.search_nsamples, args.model_seq_len, args.seed)
    total_search_tokens = sum(batch.numel() for batch in search_batches)
    log(f"Search batches ready | batches={len(search_batches)} | tokens={total_search_tokens}")
    on_policy_batches = None
    needs_on_policy_batches = args.fitness_fn == "on_policy_kl" or rerank_enabled
    if needs_on_policy_batches:
        on_policy_batches = build_on_policy_prompts(search_batches, args.on_policy_prompt_len)
        log(
            f"On-policy KL prompts ready | prompts={len(on_policy_batches)} | "
            f"prompt_len={args.on_policy_prompt_len} | rollout_len={args.on_policy_rollout_len} | "
            f"eval_every={args.on_policy_eval_every}"
        )
    if (not loaded_evo_init_cache) and args.init_parent_method == "on_policy_gradient":
        init_parent_nsamples = args.init_parent_on_policy_nsamples or args.search_nsamples
        init_parent_dataset = args.init_parent_on_policy_dataset or args.dataset
        log(
            f"Preparing {init_parent_nsamples} on-policy gradient parent prompts from {init_parent_dataset} | "
            f"prompt_len={args.on_policy_prompt_len}"
        )
        init_parent_prompts = get_prompt_loaders(
            init_parent_dataset,
            nsamples=init_parent_nsamples,
            seed=args.seed + 100000,
            prompt_len=args.on_policy_prompt_len,
            tokenizer=tokenizer,
        )
        log(f"On-policy gradient parent prompts ready | prompts={len(init_parent_prompts)}")
        warm_start_parent = parent
        warm_start_budget = budget
        if args.init_parent_warm_start_ratio is not None:
            warm_start_budget = int((1.0 - args.init_parent_warm_start_ratio) * total_dense_params)
            warm_start_ranks = initialize_ranks(spaces, warm_start_budget, args.init_strategy)
            warm_start_parent = build_topk_genome(spaces, warm_start_ranks)
            repair_budget(warm_start_parent, spaces, warm_start_budget)
        log(
            "Applying warm-start parent as student for on-policy gradient source collection | "
            f"warm_start_ratio={args.init_parent_warm_start_ratio if args.init_parent_warm_start_ratio is not None else args.ratio:.4f} | "
            f"warm_start_kept_params={total_cost(spaces, warm_start_parent['ranks'])}/{warm_start_budget} | "
            f"final_kept_params={total_cost(spaces, parent['ranks'])}/{budget}"
        )
        apply_genome(model, spaces, warm_start_parent)
        add_on_policy_gradient_parent_source(
            model,
            spaces,
            parent,
            dense_modules,
            profiling_mats,
            init_parent_prompts,
            args.DEV,
            rollout_len=args.on_policy_rollout_len,
            temperature=args.on_policy_temperature,
            eval_batch_size=args.eval_batch_size,
            lambda0=args.init_parent_lambda0,
            whitening_mode=args.init_parent_gradient_mode,
            use_kv_cache=not args.disable_init_parent_kv_cache,
            slice_kd_lm_head=not args.disable_init_parent_lm_head_slicing,
            layer_tail_ratio=args.init_parent_on_policy_layer_tail_ratio,
            input_covariance_source=args.init_parent_input_covariance_source,
            source_name="on_policy_grad",
        )
        log(
            f"On-policy gradient initial parent prepared | "
            f"kept_params={total_cost(spaces, parent['ranks'])} | "
            f"default_source={spaces[0].source_names[parent['sources'][0]] if spaces else 'n/a'}"
        )
    elif loaded_evo_init_cache and args.init_parent_method == "on_policy_gradient":
        log("Skipping on-policy gradient parent initialization because --evo_init_cache_path was loaded")
    if (not loaded_evo_init_cache) and args.save_evo_init_cache_path is not None:
        log(f"Saving evolutionary initialization cache to {args.save_evo_init_cache_path}")
        save_evo_init_cache(
            args.save_evo_init_cache_path,
            args,
            spaces,
            parent,
            total_dense_params,
            budget,
        )
        log("Saved evolutionary initialization cache")
    bulk_fitness_batches = on_policy_batches if bulk_fitness_fn == "on_policy_kl" else search_batches
    teacher_logits = None
    if fitness_requires_teacher_logits(bulk_fitness_fn) or (
        rerank_enabled and args.rerank_selection_fitness == "on_policy_plus_kl"
    ):
        if args.teacher_logits_mode == "cache":
            teacher_logits = precompute_teacher_logits(model, search_batches, args.DEV)
        else:
            log("Dense teacher logits will be streamed per KL evaluation instead of cached")
    else:
        log("Teacher logits skipped because fitness does not require them")

    log("Evaluating initial parent genome")
    parent_base_score = evaluate_genome(
        model,
        spaces,
        parent,
        bulk_fitness_batches,
        args.DEV,
        bulk_fitness_fn,
        teacher_logits,
        args.hybrid_alpha,
        args.eval_batch_size,
        dense_modules=dense_modules,
        on_policy_rollout_len=args.on_policy_rollout_len,
        on_policy_temperature=args.on_policy_temperature,
        on_policy_eval_every=args.on_policy_eval_every,
    )
    if rerank_enabled:
        parent_on_policy_score = compute_fitness(
            model,
            on_policy_batches,
            args.DEV,
            "on_policy_kl",
            teacher_logits=None,
            alpha=args.hybrid_alpha,
            eval_batch_size=args.eval_batch_size,
            spaces=spaces,
            genome=parent,
            dense_modules=dense_modules,
            on_policy_rollout_len=args.on_policy_rollout_len,
            on_policy_temperature=args.on_policy_temperature,
            on_policy_eval_every=args.on_policy_eval_every,
        )
        if args.rerank_selection_fitness == "on_policy_plus_kl":
            parent_kl_score = compute_fitness(
                model,
                search_batches,
                args.DEV,
                "kl",
                teacher_logits,
                args.hybrid_alpha,
                args.eval_batch_size,
                spaces=spaces,
                genome=parent,
                dense_modules=dense_modules,
                on_policy_rollout_len=args.on_policy_rollout_len,
                on_policy_temperature=args.on_policy_temperature,
                on_policy_eval_every=args.on_policy_eval_every,
            )
            parent_score = args.rerank_on_policy_weight * parent_on_policy_score + parent_kl_score
        else:
            parent_kl_score = None
            parent_score = parent_on_policy_score
    else:
        parent_on_policy_score = None
        parent_kl_score = None
        parent_score = parent_base_score
    best_genome = {
        "ranks": list(parent["ranks"]),
        "selected": [list(item) for item in parent["selected"]],
        "sources": list(parent["sources"]),
    }
    best_score = parent_score

    if rerank_enabled:
        if parent_kl_score is not None:
            log(
                f"Initial fitness={parent_score:.6f} | base_{bulk_fitness_fn}={parent_base_score:.6f} | "
                f"selection_on_policy_kl={parent_on_policy_score:.6f} | "
                f"selection_on_policy_weight={args.rerank_on_policy_weight:.6f} | "
                f"selection_kl={parent_kl_score:.6f} | "
                f"selection_metric={selection_fitness_name}"
            )
        else:
            log(
                f"Initial fitness={parent_score:.6f} | base_{bulk_fitness_fn}={parent_base_score:.6f} | "
                f"selection_metric={selection_fitness_name}"
            )
    else:
        log(f"Initial fitness={parent_score:.6f}")
    log(f"Initial kept params={total_cost(spaces, parent['ranks'])}/{budget}")

    for generation in range(args.generations):
        generation_start = time.time()
        log(f"Generation {generation + 1}/{args.generations} started")
        candidates = [parent]
        for _ in range(args.offspring):
            candidates.append(
                mutate_offspring(parent, spaces, args.mutation_granularity, args.max_mutations, budget)
            )
        log(f"Generated {len(candidates) - 1} offspring candidates")

        fitnesses = []
        for candidate_idx, candidate in enumerate(candidates):
            if candidate_idx == 0:
                # Parent genome remains applied from the previous step.
                rollback = None
            else:
                rollback = apply_genome_diff(model, spaces, parent, candidate)
            score = compute_fitness(
                model,
                bulk_fitness_batches,
                args.DEV,
                bulk_fitness_fn,
                teacher_logits,
                args.hybrid_alpha,
                args.eval_batch_size,
                spaces=spaces,
                genome=candidate,
                dense_modules=dense_modules,
                on_policy_rollout_len=args.on_policy_rollout_len,
                on_policy_temperature=args.on_policy_temperature,
                on_policy_eval_every=args.on_policy_eval_every,
            )
            if rollback is not None:
                rollback_modules(model, rollback)
            fitnesses.append(score)
            if candidate_idx == 0 or candidate_idx == len(candidates) - 1 or len(candidates) <= 4:
                log(
                    f"Generation {generation + 1}: evaluated candidate "
                    f"{candidate_idx + 1}/{len(candidates)} | {bulk_fitness_fn}={score:.6f}"
                )

        if rerank_enabled:
            shortlist_k = min(args.rerank_topk_on_policy, len(candidates))
            shortlist = sorted(range(len(candidates)), key=lambda idx: fitnesses[idx])[:shortlist_k]
            shortlist_scores = {}
            for candidate_idx in shortlist:
                candidate = candidates[candidate_idx]
                if candidate_idx == 0:
                    rollback = None
                else:
                    rollback = apply_genome_diff(model, spaces, parent, candidate)
                on_policy_score = compute_fitness(
                    model,
                    on_policy_batches,
                    args.DEV,
                    "on_policy_kl",
                    teacher_logits=None,
                    alpha=args.hybrid_alpha,
                    eval_batch_size=args.eval_batch_size,
                    spaces=spaces,
                    genome=candidate,
                    dense_modules=dense_modules,
                    on_policy_rollout_len=args.on_policy_rollout_len,
                    on_policy_temperature=args.on_policy_temperature,
                    on_policy_eval_every=args.on_policy_eval_every,
                )
                if args.rerank_selection_fitness == "on_policy_plus_kl":
                    kl_score = compute_fitness(
                        model,
                        search_batches,
                        args.DEV,
                        "kl",
                        teacher_logits,
                        args.hybrid_alpha,
                        args.eval_batch_size,
                        spaces=spaces,
                        genome=candidate,
                        dense_modules=dense_modules,
                        on_policy_rollout_len=args.on_policy_rollout_len,
                        on_policy_temperature=args.on_policy_temperature,
                        on_policy_eval_every=args.on_policy_eval_every,
                    )
                    final_score = args.rerank_on_policy_weight * on_policy_score + kl_score
                else:
                    kl_score = None
                    final_score = on_policy_score
                if rollback is not None:
                    rollback_modules(model, rollback)
                shortlist_scores[candidate_idx] = {
                    "final": final_score,
                    "on_policy_kl": on_policy_score,
                    "kl": kl_score,
                }
            best_idx = min(shortlist, key=lambda idx: shortlist_scores[idx]["final"])
            parent_base_score = fitnesses[best_idx]
            parent_score = shortlist_scores[best_idx]["final"]
        else:
            best_idx = min(range(len(candidates)), key=lambda idx: fitnesses[idx])
            parent_base_score = fitnesses[best_idx]
            parent_score = fitnesses[best_idx]
        previous_parent = parent
        parent = {
            "ranks": list(candidates[best_idx]["ranks"]),
            "selected": [list(item) for item in candidates[best_idx]["selected"]],
            "sources": list(candidates[best_idx]["sources"]),
        }

        if best_idx != 0:
            apply_genome_diff(model, spaces, previous_parent, parent)

        if parent_score < best_score:
            best_score = parent_score
            best_genome = {
                "ranks": list(parent["ranks"]),
                "selected": [list(item) for item in parent["selected"]],
                "sources": list(parent["sources"]),
            }
            improvement_status = "new_best"
        else:
            improvement_status = "no_global_improvement"

        if rerank_enabled and args.rerank_selection_fitness == "on_policy_plus_kl":
            log(
                f"Generation {generation + 1}/{args.generations} finished | "
                f"parent_fitness={parent_score:.6f} | best_fitness={best_score:.6f} | "
                f"base_{bulk_fitness_fn}={parent_base_score:.6f} | "
                f"selection_on_policy_kl={shortlist_scores[best_idx]['on_policy_kl']:.6f} | "
                f"selection_on_policy_weight={args.rerank_on_policy_weight:.6f} | "
                f"selection_kl={shortlist_scores[best_idx]['kl']:.6f} | "
                f"kept_params={total_cost(spaces, parent['ranks'])}/{budget} | "
                f"status={improvement_status} | elapsed={time.time() - generation_start:.1f}s"
            )
        else:
            log(
                f"Generation {generation + 1}/{args.generations} finished | "
                f"parent_fitness={parent_score:.6f} | best_fitness={best_score:.6f} | "
                f"base_{bulk_fitness_fn}={parent_base_score:.6f} | "
                f"kept_params={total_cost(spaces, parent['ranks'])}/{budget} | "
                f"status={improvement_status} | elapsed={time.time() - generation_start:.1f}s"
            )

    log("Applying best genome to the model")
    apply_genome(model, spaces, best_genome)

    if args.save_path is not None:
        os.makedirs(args.save_path, exist_ok=True)
        prefix = args.model.replace("/", "_").replace("-", "_")
        result_path = os.path.join(args.save_path, f"{prefix}_evo_svd_config.json")
        with open(result_path, "w", encoding="utf-8") as handle:
            json.dump(genome_to_serializable(spaces, best_genome), handle, indent=2)
        log(f"Saved config to {result_path}")

        # if args.profiling_mat_path is None:
        #     profiling_path = os.path.join(
        #         args.save_path,
        #         f"{prefix}_profiling_{args.dataset}_{args.whitening_nsamples}_{args.seed}.pt",
        #     )
        #     torch.save(profiling_mat, profiling_path)
        #     log(f"Saved profiling matrices to {profiling_path}")

        if args.save_model:
            model_path = os.path.join(args.save_path, f"{prefix}_evo_svd_{args.ratio}.pt")
            if is_sharded_model(model):
                log("Saving sharded compressed model without calling model.cpu()")
                torch.save({"model": model, "tokenizer": tokenizer}, model_path)
            else:
                torch.save({"model": model.cpu(), "tokenizer": tokenizer}, model_path)
            log(f"Saved compressed model to {model_path}")

    log(f"Run finished successfully in {time.time() - run_start:.1f}s")


if __name__ == "__main__":
    main()
