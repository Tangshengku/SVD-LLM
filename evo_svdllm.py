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

from SVDLLM import profle_svdllm_low_resource
from component.low_rank_linear import LowRankLinear, ZeroLinear
from utils.data_utils import get_calib_train_data, get_loaders
from utils.model_utils import find_layers, get_model_from_huggingface


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


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


def clone_bias(module: nn.Linear) -> Optional[torch.Tensor]:
    if module.bias is None:
        return None
    return module.bias.detach().cpu().clone()


def parse_source_datasets(spec: str) -> List[str]:
    datasets = [item.strip() for item in spec.split(",") if item.strip()]
    if not datasets:
        raise ValueError("At least one source dataset must be provided.")
    return datasets


def build_source_profile_plan(source_datasets: Sequence[str]) -> List[Tuple[str, str]]:
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
        # lambda genome: mutate_rank_transfer(genome, spaces, mutation_granularity),
        # lambda genome: mutate_boundary_swap(genome, spaces),
        lambda genome: mutate_source_choice(genome, spaces),
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
        chunk = chunk.to(device)                          # [bs, seqlen]
        logits = model(chunk, use_cache=False).logits.float()  # fp32 prevents log_softmax underflow
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = chunk[:, 1:].contiguous()
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
        logits = model(batch.to(device), use_cache=False).logits[:, :-1, :].float()
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
    teacher_logits: Sequence[torch.Tensor],
    device: str,
    eval_batch_size: int = 1,
) -> float:
    model.eval()
    total_loss = 0.0
    total_seqs = 0
    for chunk, teacher_chunk in tqdm(zip(
        _iter_minibatches(batches, eval_batch_size),
        _iter_minibatches(teacher_logits, eval_batch_size)), desc="computing kl"
    ):
        chunk = chunk.to(device)
        student_logits = model(chunk, use_cache=False).logits[:, :-1, :].float()
        if not torch.isfinite(student_logits).all():
            raise RuntimeError("Non-finite student logits detected during KL computation")
        teacher_log_prob = torch.log_softmax(teacher_chunk.to(device), dim=-1)
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
def compute_fitness(
    model,
    batches: Sequence[torch.Tensor],
    device: str,
    fitness_fn: str,
    teacher_logits: Optional[Sequence[torch.Tensor]] = None,
    alpha: float = 0.5,
    eval_batch_size: int = 1,
) -> float:
    """Evaluate fitness of the model in its current state (genome already applied)."""
    if fitness_fn == "ppl":
        return math.exp(compute_nll(model, batches, device, eval_batch_size))
    if fitness_fn == "kl":
        return compute_kl(model, batches, teacher_logits, device, eval_batch_size)
    nll = compute_nll(model, batches, device, eval_batch_size)
    kl = compute_kl(model, batches, teacher_logits, device, eval_batch_size)
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
) -> float:
    apply_genome(model, spaces, genome)
    return compute_fitness(model, batches, device, fitness_fn, teacher_logits, alpha, eval_batch_size)


def get_search_batches(dataset: str, tokenizer, nsamples: int, seqlen: int, seed: int) -> List[torch.Tensor]:
    loader, _ = get_loaders(dataset, nsamples=nsamples, seed=seed, tokenizer=tokenizer, seqlen=seqlen)
    return [inp for inp, _ in loader]


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
        help="Comma-separated datasets used to build source-specific whitening profiles. When multiple datasets are given, an additional mixed source is built automatically and used for parent initialization.",
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
    parser.add_argument("--save_path", type=str, default=None, help="Directory for configs or model checkpoints.")
    parser.add_argument("--save_model", action="store_true", help="Save the final compressed model checkpoint.")
    parser.add_argument("--fitness_fn", choices=["ppl", "kl", "hyb"], default="kl", help="Search fitness.")
    parser.add_argument("--hybrid_alpha", type=float, default=0.5, help="Weight of KL in hybrid fitness.")
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
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--DEV", type=str, default="cuda", help="Search device.")
    parser.add_argument("--eval_batch_size", type=int, default=4, help="Mini-batch size for fitness forward passes.")
    return parser.parse_args()


def main():
    args = parse_args()
    source_datasets = parse_source_datasets(args.source_datasets)
    source_profile_plan = build_source_profile_plan(source_datasets)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    run_start = time.time()

    log(
        f"Launching evolutionary SVD search | model={args.model} | ratio={args.ratio} | "
        f"dataset={args.dataset} | fitness={args.fitness_fn} | generations={args.generations} | "
        f"offspring={args.offspring} | sources={source_datasets} | "
        f"profile_plan={[name for name, _ in source_profile_plan]} | device={args.DEV}"
    )

    log("Loading dense model and tokenizer")
    model, tokenizer = get_model_from_huggingface(args.model)
    model.eval()
    model.seqlen = args.model_seq_len
    model = model.to(args.DEV)
    model.config.use_cache = False
    log(f"Loaded model; sequence length set to {args.model_seq_len}")

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

    # The low-resource profiling path moves major submodules back to CPU.
    # Move the full dense model to the target device again before search-time evaluation.
    model = model.to(args.DEV)
    model.eval()
    model.config.use_cache = False
    log(f"Dense model moved back to {args.DEV} for search-time evaluation")

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
    teacher_logits = None
    if args.fitness_fn in ["kl", "hyb"]:
        teacher_logits = precompute_teacher_logits(model, search_batches, args.DEV)
    else:
        log("Teacher logits skipped because fitness does not require them")

    log("Evaluating initial parent genome")
    parent_score = evaluate_genome(
        model,
        spaces,
        parent,
        search_batches,
        args.DEV,
        args.fitness_fn,
        teacher_logits,
        args.hybrid_alpha,
        args.eval_batch_size,
    )
    best_genome = {
        "ranks": list(parent["ranks"]),
        "selected": [list(item) for item in parent["selected"]],
        "sources": list(parent["sources"]),
    }
    best_score = parent_score

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
                search_batches,
                args.DEV,
                args.fitness_fn,
                teacher_logits,
                args.hybrid_alpha,
                args.eval_batch_size,
            )
            if rollback is not None:
                rollback_modules(model, rollback)
            fitnesses.append(score)
            if candidate_idx == 0 or candidate_idx == len(candidates) - 1 or len(candidates) <= 4:
                log(
                    f"Generation {generation + 1}: evaluated candidate "
                    f"{candidate_idx + 1}/{len(candidates)} | fitness={score:.6f}"
                )

        best_idx = min(range(len(candidates)), key=lambda idx: fitnesses[idx])
        previous_parent = parent
        parent = {
            "ranks": list(candidates[best_idx]["ranks"]),
            "selected": [list(item) for item in candidates[best_idx]["selected"]],
            "sources": list(candidates[best_idx]["sources"]),
        }
        parent_score = fitnesses[best_idx]

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

        log(
            f"Generation {generation + 1}/{args.generations} finished | "
            f"parent_fitness={parent_score:.6f} | best_fitness={best_score:.6f} | "
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
            torch.save({"model": model.cpu(), "tokenizer": tokenizer}, model_path)
            log(f"Saved compressed model to {model_path}")

    log(f"Run finished successfully in {time.time() - run_start:.1f}s")


if __name__ == "__main__":
    main()
