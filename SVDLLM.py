#coding:utf8
import os
import sys
import argparse
import math
import torch.jit
from tqdm import tqdm
import torch
import torch.nn as nn

from utils.data_utils import *
from component.svd_llama import SVD_LlamaAttention, SVD_LlamaMLP
from component.svd_mistral import SVD_MistralAttention, SVD_MistralMLP
from component.svd_qwen3 import SVD_Qwen3Attention, SVD_Qwen3MLP
from component.svd_opt import SVDOPTDecoderLayer
from utils.model_utils import *
from evaluater import * 

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)


def _move_nested_to_cpu(value):
    if torch.is_tensor(value):
        return value.cpu()
    if isinstance(value, tuple):
        return tuple(_move_nested_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [_move_nested_to_cpu(item) for item in value]
    if isinstance(value, dict):
        return {key: _move_nested_to_cpu(item) for key, item in value.items()}
    return value


def _move_nested_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move_nested_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_nested_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_nested_to_device(item, device) for key, item in value.items()}
    return value


def _stack_nested(values):
    first = values[0]
    if torch.is_tensor(first):
        return torch.cat(values, dim=0)
    if isinstance(first, tuple):
        return tuple(_stack_nested([item[idx] for item in values]) for idx in range(len(first)))
    if isinstance(first, list):
        return [_stack_nested([item[idx] for item in values]) for idx in range(len(first))]
    if isinstance(first, dict):
        return {key: _stack_nested([item[key] for item in values]) for key in first}
    return first


def set_submodule(root: nn.Module, path: str, module: nn.Module) -> None:
    if "." in path:
        parent_path, attr_name = path.rsplit(".", 1)
        parent = root.get_submodule(parent_path)
    else:
        parent = root
        attr_name = path
    setattr(parent, attr_name, module)


def _iter_tensor_batches(samples, batch_size):
    for start in range(0, len(samples), batch_size):
        yield torch.cat(list(samples[start : start + batch_size]), dim=0)


def log(message: str) -> None:
    print(message, flush=True)


def _prepare_model_for_pickle(model):
    if not isinstance(model, nn.Module):
        return model
    try:
        from accelerate.hooks import remove_hook_from_submodules
    except ImportError:
        remove_hook_from_submodules = None
    if remove_hook_from_submodules is not None:
        remove_hook_from_submodules(model)
    for module in model.modules():
        old_forward = getattr(module, "_old_forward", None)
        if old_forward is not None:
            module.forward = old_forward
            try:
                delattr(module, "_old_forward")
            except AttributeError:
                pass
        for attr_name in ("_hf_hook", "_hf_hooks"):
            if hasattr(module, attr_name):
                try:
                    delattr(module, attr_name)
                except AttributeError:
                    pass
    return model


def save_model_checkpoint(model, tokenizer, path):
    model = _prepare_model_for_pickle(model)
    torch.save({'model': model, 'tokenizer': tokenizer}, path)


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


def _is_float8_dtype(dtype):
    return dtype in FLOAT8_DTYPES


def _runtime_svd_dtype(dtype):
    return torch.float16 if _is_float8_dtype(dtype) else dtype


def _parse_svd_output_dtype(name):
    if name in (None, "original"):
        return None
    if name == "runtime":
        return "runtime"
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    if name == "float8_e4m3fn":
        if not hasattr(torch, "float8_e4m3fn"):
            raise ValueError("This PyTorch build does not expose torch.float8_e4m3fn")
        return torch.float8_e4m3fn
    if name == "float8_e5m2":
        if not hasattr(torch, "float8_e5m2"):
            raise ValueError("This PyTorch build does not expose torch.float8_e5m2")
        return torch.float8_e5m2
    raise ValueError(f"Unsupported --svd_output_dtype: {name}")


def _resolve_output_dtype(source_dtype, output_dtype_policy):
    if output_dtype_policy is None:
        return source_dtype
    if output_dtype_policy == "runtime":
        return _runtime_svd_dtype(source_dtype)
    return output_dtype_policy


def _low_rank_from_ratio(rows, cols, ratio):
    full_rank = min(rows, cols)
    rank = int(rows * cols * ratio / (rows + cols))
    return max(1, min(full_rank, rank))


def _factor_group_name(name):
    leaf = name.rsplit(".", 1)[-1]
    for suffix in ("q_proj", "k_proj", "v_proj", "o_proj", "out_proj", "gate_proj", "down_proj", "up_proj", "fc1", "fc2"):
        if suffix in leaf or suffix in name:
            return suffix
    return leaf


def _ensure_factor_pair(owner, u_attr, v_attr, svd_u, svd_v):
    u_proj = getattr(owner, u_attr)
    v_proj = getattr(owner, v_attr)
    out_features, rank = svd_u.shape
    v_rank, in_features = svd_v.shape
    if rank != v_rank:
        raise ValueError(f"Mismatched factor ranks for {u_attr}/{v_attr}: {rank} vs {v_rank}")
    if (
        u_proj.weight.shape == svd_u.shape
        and v_proj.weight.shape == svd_v.shape
    ):
        return
    u_bias = u_proj.bias is not None
    v_bias = v_proj.bias is not None
    new_u = nn.Linear(rank, out_features, bias=u_bias)
    new_v = nn.Linear(in_features, rank, bias=v_bias)
    setattr(owner, u_attr, new_u)
    setattr(owner, v_attr, new_v)


def _dynamic_rank_allocation(model_name, model, profiling_mat, ratio, dev):
    model_name = model_name.lower()
    if 'opt' in model_name:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers

    entries = []
    print("Computing SVD-LLM V2 layer sensitivity for dynamic rank allocation...")
    for i in tqdm(range(len(layers))):
        layer = layers[i]
        subset = _resolve_profiled_layers(layer, profiling_mat[i])
        for name in subset:
            scaling_diag_matrix = profiling_mat[i][name].to(dev)
            weight_tensor = _get_linear_like_weight(subset[name], expected_in_features=scaling_diag_matrix.shape[0])
            if weight_tensor is None:
                continue
            W = weight_tensor.detach().float().to(dev)
            rows, cols = W.shape
            try:
                W_scale = torch.matmul(W, scaling_diag_matrix.float())
                _, S, _ = torch.linalg.svd(W_scale, full_matrices=False)
            except torch.OutOfMemoryError:
                del W, scaling_diag_matrix
                torch.cuda.empty_cache()
                raise
            base_rank = _low_rank_from_ratio(rows, cols, ratio)
            tail = S[base_rank:]
            loss = torch.linalg.vector_norm(tail).item() if tail.numel() else 0.0
            entries.append(
                {
                    "key": (i, name),
                    "group": _factor_group_name(name),
                    "rows": rows,
                    "cols": cols,
                    "base_rank": base_rank,
                    "full_rank": min(rows, cols),
                    "loss": max(loss, 0.0),
                }
            )
            del W, W_scale, S, scaling_diag_matrix, tail
            torch.cuda.empty_cache()

    target_compression = max(0.0, min(1.0, 1.0 - ratio))
    by_group = {}
    for entry in entries:
        by_group.setdefault(entry["group"], []).append(entry)

    rank_map = {}
    ratio_map = {}
    for group_entries in by_group.values():
        target_rank_cost = sum(
            entry["base_rank"] * (entry["rows"] + entry["cols"])
            for entry in group_entries
        )
        losses = [entry["loss"] for entry in group_entries]
        loss_mean = sum(losses) / max(len(losses), 1)
        loss_mean = max(loss_mean, 1e-12)
        scores = []
        for entry in group_entries:
            # SVD-LLM V2 applies inverse-log normalization to theoretical
            # truncation loss. Normalize by the group mean first so tiny raw
            # losses do not dominate and collapse ranks to one.
            normalized_loss = entry["loss"] / loss_mean
            score = 1.0 / max(math.log(math.e + normalized_loss), 1e-12)
            scores.append(score)
        score_sum = sum(scores)
        for entry, score in zip(group_entries, scores):
            allocated_compression = len(group_entries) * target_compression * score / score_sum if score_sum > 0 else target_compression
            allocated_compression = max(0.0, min(0.95, allocated_compression))
            allocated_ratio = 1.0 - allocated_compression
            rank = _low_rank_from_ratio(entry["rows"], entry["cols"], allocated_ratio)
            rank_map[entry["key"]] = rank
            ratio_map[entry["key"]] = allocated_ratio

        def current_rank_cost():
            return sum(
                rank_map[entry["key"]] * (entry["rows"] + entry["cols"])
                for entry in group_entries
            )

        # Integer ranks and clipping can drift away from the requested budget.
        # Repair the group to the same low-rank parameter budget as fixed-ratio
        # SVD-LLM, removing rank from low-sensitivity weights first and adding it
        # back to high-sensitivity weights first.
        while current_rank_cost() > target_rank_cost:
            candidates = [
                (entry["loss"], entry)
                for entry in group_entries
                if rank_map[entry["key"]] > 1
            ]
            if not candidates:
                break
            _, entry = min(candidates, key=lambda item: item[0])
            rank_map[entry["key"]] -= 1
        while current_rank_cost() < target_rank_cost:
            candidates = [
                (entry["loss"], entry)
                for entry in group_entries
                if rank_map[entry["key"]] < entry["full_rank"]
            ]
            if not candidates:
                break
            _, entry = max(candidates, key=lambda item: item[0])
            rank_map[entry["key"]] += 1
        for entry in group_entries:
            rank = rank_map[entry["key"]]
            ratio_map[entry["key"]] = rank * (entry["rows"] + entry["cols"]) / max(entry["rows"] * entry["cols"], 1)
        group_name = group_entries[0]["group"]
        group_ranks = [rank_map[entry["key"]] for entry in group_entries]
        group_cost = current_rank_cost()
        print(
            f"Dynamic allocation group={group_name}: "
            f"weights={len(group_entries)}, rank_range=[{min(group_ranks)}, {max(group_ranks)}], "
            f"avg_rank={sum(group_ranks) / len(group_ranks):.1f}, "
            f"rank_cost={group_cost}/{target_rank_cost}"
        )

    if rank_map:
        avg_ratio = sum(ratio_map.values()) / len(ratio_map)
        min_rank = min(rank_map.values())
        max_rank = max(rank_map.values())
        print(
            f"Dynamic rank allocation ready for {len(rank_map)} weights "
            f"(target_keep_ratio={ratio:.4f}, avg_allocated_keep_ratio={avg_ratio:.4f}, "
            f"rank_range=[{min_rank}, {max_rank}])"
        )
    return rank_map, ratio_map


def _is_linear_like_module(module):
    weight = getattr(module, "weight", None)
    return isinstance(weight, torch.Tensor) and weight.ndim == 2


def _get_own_linear_like_weight(module):
    weight = getattr(module, "weight", None)
    if isinstance(weight, torch.Tensor) and weight.ndim == 2 and weight.is_floating_point():
        return weight
    for _, tensor in module.named_parameters(recurse=False):
        if isinstance(tensor, torch.Tensor) and tensor.ndim == 2 and tensor.is_floating_point():
            return tensor
    for _, tensor in module.named_buffers(recurse=False):
        if isinstance(tensor, torch.Tensor) and tensor.ndim == 2 and tensor.is_floating_point():
            return tensor
    return None


def _get_linear_like_weight(module, expected_in_features=None):
    candidates = []
    weight = getattr(module, "weight", None)
    if isinstance(weight, torch.Tensor) and weight.ndim == 2 and weight.is_floating_point():
        candidates.append(("weight", weight))
    for name, tensor in module.named_parameters(recurse=True):
        if isinstance(tensor, torch.Tensor) and tensor.ndim == 2 and tensor.is_floating_point():
            candidates.append((name, tensor))
    for name, tensor in module.named_buffers(recurse=True):
        if isinstance(tensor, torch.Tensor) and tensor.ndim == 2 and tensor.is_floating_point():
            candidates.append((name, tensor))
    if not candidates:
        return None
    if expected_in_features is not None:
        for _, tensor in candidates:
            if tensor.shape[1] == expected_in_features:
                return tensor
    return candidates[0][1]


def _find_linear_like_layers(module, name=""):
    if _get_own_linear_like_weight(module) is not None:
        return {name: module}
    res = {}
    for child_name, child in module.named_children():
        child_path = name + "." + child_name if name else child_name
        res.update(_find_linear_like_layers(child, child_path))
    return res


def _resolve_profiled_layers(layer, layer_profile=None):
    if layer_profile is None:
        return _find_linear_like_layers(layer)
    subset = {}
    for local_name, factor in layer_profile.items():
        try:
            module = layer.get_submodule(local_name)
        except AttributeError:
            continue
        expected_in = factor.shape[0] if torch.is_tensor(factor) and factor.ndim >= 2 else None
        if _get_linear_like_weight(module, expected_in_features=expected_in) is not None:
            subset[local_name] = module
    return subset


def _clone_optional_bias(module, dtype=None):
    bias = getattr(module, "bias", None)
    if bias is None:
        return None
    if dtype is not None and not _is_float8_dtype(dtype):
        return bias.detach().cpu().to(dtype)
    return bias.detach().cpu()


def _module_device(module, fallback="cpu"):
    for tensor in module.parameters(recurse=True):
        if tensor.device.type != "meta":
            return tensor.device
    for tensor in module.buffers(recurse=True):
        if tensor.device.type != "meta":
            return tensor.device
    return torch.device(fallback)


def _has_accelerate_device_map(model):
    return hasattr(model, "hf_device_map") and getattr(model, "hf_device_map")


def _sample_top_p(logits, temperature=1.0, top_p=1.0):
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


def _cholesky_with_jitter(matrix, device, dtype=torch.float64):
    work = matrix.to(device=device, dtype=dtype)
    work = torch.nan_to_num(work, nan=0.0, posinf=0.0, neginf=0.0)
    work = 0.5 * (work + work.transpose(0, 1))
    eye = torch.eye(work.shape[0], device=device, dtype=dtype)
    scale = work.diagonal().abs().mean().item()
    scale = max(scale, 1.0)
    jitter = 0.0
    for _ in range(8):
        try:
            return torch.linalg.cholesky(work + jitter * eye)
        except Exception:
            jitter = scale * 1e-6 if jitter == 0.0 else jitter * 10
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(work)
    except Exception:
        cpu_work = work.cpu()
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(cpu_work)
            eigenvalues = eigenvalues.to(device)
            eigenvectors = eigenvectors.to(device)
        except Exception:
            diag = torch.clamp(work.diagonal(), min=scale * 1e-6)
            return torch.diag(torch.sqrt(diag))
    eigenvalues = torch.clamp(eigenvalues, min=scale * 1e-6)
    return eigenvectors.matmul(torch.diag(torch.sqrt(eigenvalues)))


def _capture_module_snapshot(model, module_paths):
    return [(path, model.get_submodule(path)) for path in module_paths]


def _apply_module_snapshot(model, snapshot, device=None, offload_replaced_to_cpu=False):
    for path, module in snapshot:
        if offload_replaced_to_cpu:
            try:
                current = model.get_submodule(path)
                if current is not module:
                    current.cpu()
            except AttributeError:
                pass
        if device is not None:
            module = module.to(device)
        set_submodule(model, path, module)


def _build_replaced_module_paths(model_name, model):
    model_name = model_name.lower()
    if "opt" in model_name:
        return [f"model.decoder.layers.{idx}" for idx in range(len(model.model.decoder.layers))]
    return [
        f"model.layers.{idx}.self_attn" for idx in range(len(model.model.layers))
    ] + [
        f"model.layers.{idx}.mlp" for idx in range(len(model.model.layers))
    ]


def _build_low_rank_hook_specs(model_name, model):
    model_name = model_name.lower()
    specs = []

    def add_spec_if_present(layer_idx, logical_name, input_path, output_path):
        try:
            model.get_submodule(input_path)
            model.get_submodule(output_path)
        except AttributeError:
            return
        specs.append((layer_idx, logical_name, input_path, output_path))

    if "opt" in model_name:
        for idx in range(len(model.model.decoder.layers)):
            base = f"model.decoder.layers.{idx}"
            add_spec_if_present(idx, "self_attn.q_proj", f"{base}.self_attn.q_v_proj", f"{base}.self_attn.q_u_proj")
            add_spec_if_present(idx, "self_attn.k_proj", f"{base}.self_attn.k_v_proj", f"{base}.self_attn.k_u_proj")
            add_spec_if_present(idx, "self_attn.v_proj", f"{base}.self_attn.v_v_proj", f"{base}.self_attn.v_u_proj")
            add_spec_if_present(idx, "self_attn.out_proj", f"{base}.self_attn.out_v_proj", f"{base}.self_attn.out_u_proj")
            add_spec_if_present(idx, "fc1", f"{base}.fc1_v_proj", f"{base}.fc1_u_proj")
            add_spec_if_present(idx, "fc2", f"{base}.fc2_v_proj", f"{base}.fc2_u_proj")
            add_spec_if_present(idx, "self_attn.q_proj", f"{base}.self_attn.q_proj", f"{base}.self_attn.q_proj")
            add_spec_if_present(idx, "self_attn.k_proj", f"{base}.self_attn.k_proj", f"{base}.self_attn.k_proj")
            add_spec_if_present(idx, "self_attn.v_proj", f"{base}.self_attn.v_proj", f"{base}.self_attn.v_proj")
            add_spec_if_present(idx, "self_attn.out_proj", f"{base}.self_attn.out_proj", f"{base}.self_attn.out_proj")
            add_spec_if_present(idx, "fc1", f"{base}.fc1", f"{base}.fc1")
            add_spec_if_present(idx, "fc2", f"{base}.fc2", f"{base}.fc2")
        return specs

    for idx in range(len(model.model.layers)):
        base = f"model.layers.{idx}"
        add_spec_if_present(idx, "self_attn.q_proj", f"{base}.self_attn.q_v_proj", f"{base}.self_attn.q_u_proj")
        add_spec_if_present(idx, "self_attn.k_proj", f"{base}.self_attn.k_v_proj", f"{base}.self_attn.k_u_proj")
        add_spec_if_present(idx, "self_attn.v_proj", f"{base}.self_attn.v_v_proj", f"{base}.self_attn.v_u_proj")
        add_spec_if_present(idx, "self_attn.o_proj", f"{base}.self_attn.o_v_proj", f"{base}.self_attn.o_u_proj")
        add_spec_if_present(idx, "mlp.gate_proj", f"{base}.mlp.gate_v_proj", f"{base}.mlp.gate_u_proj")
        add_spec_if_present(idx, "mlp.down_proj", f"{base}.mlp.down_v_proj", f"{base}.mlp.down_u_proj")
        add_spec_if_present(idx, "mlp.up_proj", f"{base}.mlp.up_v_proj", f"{base}.mlp.up_u_proj")
        add_spec_if_present(idx, "self_attn.q_proj", f"{base}.self_attn.q_proj", f"{base}.self_attn.q_proj")
        add_spec_if_present(idx, "self_attn.k_proj", f"{base}.self_attn.k_proj", f"{base}.self_attn.k_proj")
        add_spec_if_present(idx, "self_attn.v_proj", f"{base}.self_attn.v_proj", f"{base}.self_attn.v_proj")
        add_spec_if_present(idx, "self_attn.o_proj", f"{base}.self_attn.o_proj", f"{base}.self_attn.o_proj")
        add_spec_if_present(idx, "mlp.gate_proj", f"{base}.mlp.gate_proj", f"{base}.mlp.gate_proj")
        add_spec_if_present(idx, "mlp.down_proj", f"{base}.mlp.down_proj", f"{base}.mlp.down_proj")
        add_spec_if_present(idx, "mlp.up_proj", f"{base}.mlp.up_proj", f"{base}.mlp.up_proj")
    return specs


class _OnPolicyCovCollector:
    def __init__(self, model, hook_specs, alpha_min, alpha_max, delta, device):
        self.model = model
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.delta = delta
        self.enabled = False
        self.device = device
        self.stats = {}
        self.handles = []
        for layer_idx, name, input_path, output_path in hook_specs:
            input_module = model.get_submodule(input_path)
            output_module = model.get_submodule(output_path)
            key = (layer_idx, name)
            self.stats[key] = {
                "cov_x": torch.zeros(
                    (input_module.in_features, input_module.in_features),
                    dtype=torch.float32,
                    device="cpu",
                ),
                "cov_g": torch.zeros(
                    (output_module.out_features, output_module.out_features),
                    dtype=torch.float32,
                    device="cpu",
                ),
                "count": 0,
                "last_input": None,
                "last_grad": None,
            }
            self.handles.append(input_module.register_forward_pre_hook(self._make_input_hook(key)))
            self.handles.append(output_module.register_full_backward_hook(self._make_grad_hook(key)))

    def _make_input_hook(self, key):
        def hook(module, inputs):
            if self.enabled:
                self.stats[key]["last_input"] = inputs[0].detach().float()
        return hook

    def _make_grad_hook(self, key):
        def hook(module, grad_input, grad_output):
            if not self.enabled:
                return
            grad = grad_output[0]
            if grad is not None:
                self.stats[key]["last_grad"] = grad.detach().float()
        return hook

    def consume_batch(self, generated_slice):
        start, end = generated_slice
        for stat in self.stats.values():
            inputs = stat["last_input"]
            grads = stat["last_grad"]
            stat["last_input"] = None
            stat["last_grad"] = None
            if inputs is None or grads is None:
                continue
            inputs = inputs[:, start:end, :]
            grads = grads[:, start:end, :]
            if inputs.numel() == 0 or grads.numel() == 0:
                continue
            flat_inputs = inputs.reshape(-1, inputs.shape[-1])
            flat_grads = grads.reshape(-1, grads.shape[-1])
            stat["cov_x"] += flat_inputs.transpose(0, 1).matmul(flat_inputs).cpu()
            stat["cov_g"] += flat_grads.transpose(0, 1).matmul(flat_grads).cpu()
            stat["count"] += flat_inputs.shape[0]
            del inputs, grads, flat_inputs, flat_grads

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def _sample_on_policy_sequences_no_cache(model, prompts, rollout_len, generation_temperature, generation_top_p):
    generated = prompts
    for _ in range(rollout_len):
        logits = model(generated, use_cache=False).logits[:, -1, :].float()
        next_token = _sample_top_p(
            logits,
            temperature=generation_temperature,
            top_p=generation_top_p,
        )
        generated = torch.cat((generated, next_token), dim=1)
    return generated


def _sample_on_policy_sequences(model, prompts, rollout_len, generation_temperature, generation_top_p, use_kv_cache=True):
    if not use_kv_cache:
        return _sample_on_policy_sequences_no_cache(
            model,
            prompts,
            rollout_len=rollout_len,
            generation_temperature=generation_temperature,
            generation_top_p=generation_top_p,
        )
    generated = prompts
    try:
        outputs = model(prompts, use_cache=True)
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :].float()
        for step_idx in range(rollout_len):
            next_token = _sample_top_p(
                logits,
                temperature=generation_temperature,
                top_p=generation_top_p,
            )
            generated = torch.cat((generated, next_token), dim=1)
            if step_idx == rollout_len - 1:
                break
            outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :].float()
        return generated
    except Exception as err:
        log(f"KV-cache rollout failed ({type(err).__name__}: {err}); falling back to no-cache rollout")
        return _sample_on_policy_sequences_no_cache(
            model,
            prompts,
            rollout_len=rollout_len,
            generation_temperature=generation_temperature,
            generation_top_p=generation_top_p,
        )


def _compute_reverse_kd_loss(student_logits, teacher_logits, prompt_len, kd_temperature):
    if prompt_len <= 0:
        raise ValueError("prompt_len must be positive")
    student_slice = student_logits[:, prompt_len - 1 : -1, :].float()
    teacher_slice = teacher_logits[:, prompt_len - 1 : -1, :].float()
    student_log_prob = torch.log_softmax(student_slice / kd_temperature, dim=-1)
    teacher_log_prob = torch.log_softmax(teacher_slice / kd_temperature, dim=-1)
    return (kd_temperature ** 2) * torch.nn.functional.kl_div(
        teacher_log_prob,
        student_log_prob,
        reduction="batchmean",
        log_target=True,
    )


def _compute_forward_kd_loss(student_logits, teacher_logits, start, end, kd_temperature):
    if end <= start:
        raise ValueError("KD loss slice must be non-empty")
    student_slice = student_logits[:, start:end, :].float()
    teacher_slice = teacher_logits[:, start:end, :].float()
    student_log_prob = torch.log_softmax(student_slice / kd_temperature, dim=-1)
    teacher_log_prob = torch.log_softmax(teacher_slice / kd_temperature, dim=-1)
    return (kd_temperature ** 2) * torch.nn.functional.kl_div(
        student_log_prob,
        teacher_log_prob,
        reduction="batchmean",
        log_target=True,
    )


def _compute_on_policy_objective(
    loss_type,
    sequences,
    student_logits,
    teacher_logits,
    prompt_len,
    rollout_len,
    kd_temperature,
):
    if loss_type == "reverse_kd":
        if teacher_logits is None:
            raise ValueError("reverse_kd requires teacher logits")
        return (
            _compute_reverse_kd_loss(student_logits, teacher_logits, prompt_len, kd_temperature),
            (prompt_len - 1, prompt_len - 1 + rollout_len),
            "reverse_kd_loss",
        )
    if loss_type == "lm":
        student_slice = student_logits[:, : prompt_len - 1, :].float()
        labels = sequences[:, 1:prompt_len].contiguous()
        loss = torch.nn.functional.cross_entropy(
            student_slice.reshape(-1, student_slice.shape[-1]),
            labels.reshape(-1),
        )
        return loss, (0, prompt_len - 1), "prefix_lm_loss"
    if loss_type == "prefix_kd":
        if teacher_logits is None:
            raise ValueError("prefix_kd requires teacher logits")
        return (
            _compute_forward_kd_loss(
                student_logits,
                teacher_logits,
                start=0,
                end=prompt_len - 1,
                kd_temperature=kd_temperature,
            ),
            (0, prompt_len - 1),
            "prefix_kd_loss",
        )
    raise ValueError(f"Unsupported on-policy loss: {loss_type}")


def _build_gradient_whitening_profile(on_stats, lambda0, dev, whitening_mode="both", offline_profile=None):
    if whitening_mode not in {"both", "grad_only"}:
        raise ValueError("whitening_mode must be one of: both, grad_only")
    profile = {}
    active_layers = 0
    total_layers = 0
    for (layer_idx, name), stat in on_stats.items():
        total_layers += 1
        layer_profile = profile.setdefault(layer_idx, {})
        if stat["count"] <= 0:
            layer_profile[name] = {
                "x": (
                    torch.eye(stat["cov_x"].shape[0], dtype=torch.float32)
                    if whitening_mode == "grad_only"
                    else (
                        offline_profile[layer_idx][name].float().cpu()
                        if offline_profile is not None
                        else torch.eye(stat["cov_x"].shape[0], dtype=torch.float32)
                    )
                ),
                "g": torch.eye(stat["cov_g"].shape[0], dtype=torch.float32),
            }
            continue
        active_layers += 1
        cov_g = stat["cov_g"].to(dev) / stat["count"]
        eps_g = lambda0 * cov_g.trace().item() / max(cov_g.shape[0], 1)
        cov_g = cov_g + eps_g * torch.eye(cov_g.shape[0], dtype=torch.float32, device=dev)
        if whitening_mode == "grad_only":
            x_factor = torch.eye(stat["cov_x"].shape[0], dtype=torch.float32)
        elif offline_profile is not None:
            x_factor = offline_profile[layer_idx][name].float().cpu()
        else:
            cov_x = stat["cov_x"].to(dev) / stat["count"]
            eps_x = lambda0 * cov_x.trace().item() / max(cov_x.shape[0], 1)
            cov_x = cov_x + eps_x * torch.eye(cov_x.shape[0], dtype=torch.float32, device=dev)
            x_factor = _cholesky_with_jitter(cov_x, dev, dtype=torch.float64).cpu()
            del cov_x
        layer_profile[name] = {
            "x": x_factor,
            "g": _cholesky_with_jitter(cov_g, dev, dtype=torch.float64).cpu(),
        }
        del cov_g
        torch.cuda.empty_cache()
    log(
        f"Built gradient whitening profile | active_layers={active_layers}/{total_layers} | "
        f"lambda0={lambda0} | whitening_mode={whitening_mode} | "
        f"input_covariance={'identity' if whitening_mode == 'grad_only' else ('offline' if offline_profile is not None else 'on_policy')} | "
        "gradient_covariance=on_policy"
    )
    return profile


@torch.no_grad()
def build_prompt_batches(dataset_name, tokenizer, nsamples, prompt_len, seqlen, seed, batch_size):
    if prompt_len <= 0 or prompt_len >= seqlen:
        raise ValueError("prompt_len must be in [1, seqlen-1]")
    log(
        f"Loading on-policy prompt dataset | dataset={dataset_name} | nsamples={nsamples} | "
        f"prompt_len={prompt_len} | seed={seed}"
    )
    prompt_samples = [
        prompt.contiguous()
        for prompt in get_prompt_loaders(
            dataset_name,
            nsamples=nsamples,
            seed=seed,
            tokenizer=tokenizer,
            prompt_len=prompt_len,
        )
    ]
    return list(_iter_tensor_batches(prompt_samples, batch_size))


def on_policy_reverse_kd_guided_whitening(
    model_name,
    model,
    tokenizer,
    ratio,
    warm_start_ratio,
    dev,
    offline_profile,
    prompt_dataset,
    prompt_nsamples,
    prompt_len,
    rollout_len,
    kd_temperature=2.0,
    generation_temperature=0.7,
    generation_top_p=0.9,
    rho=0.3,
    alpha_min=0.1,
    alpha_max=10.0,
    alpha_delta=1e-6,
    lambda0=1e-6,
    rounds=1,
    eval_batch_size=1,
    seed=0,
    init_scheme="uniform",
    gradient_whitening_mode="both",
    use_on_policy_kv_cache=True,
    on_policy_layer_tail_ratio=1.0,
    output_dtype_policy=None,
    on_policy_loss="reverse_kd",
):
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    if rollout_len <= 0:
        raise ValueError("rollout_len must be positive")
    if not (0.0 <= on_policy_layer_tail_ratio <= 1.0):
        raise ValueError("on_policy_layer_tail_ratio must be in [0, 1]")
    if on_policy_loss not in {"reverse_kd", "lm", "prefix_kd"}:
        raise ValueError("on_policy_loss must be one of: reverse_kd, lm, prefix_kd")

    model_name = model_name.lower()
    model.eval()
    model.config.use_cache = False
    log(
        f"Starting guided gradient whitening | offline_profile_layers={len(offline_profile)} | "
        f"prompt_dataset={prompt_dataset} | prompt_nsamples={prompt_nsamples} | prompt_len={prompt_len} | "
        f"rollout_len={rollout_len} | rounds={rounds} | kd_temperature={kd_temperature} | "
        f"generation_temperature={generation_temperature} | generation_top_p={generation_top_p} | "
        f"on_policy_loss={on_policy_loss}"
    )
    dense_paths = _build_replaced_module_paths(model_name, model)
    dense_snapshot = _capture_module_snapshot(model, dense_paths)

    if warm_start_ratio is None:
        warm_start_ratio = ratio
    log(
        f"Applying initial offline whitening profile to build the first student | "
        f"warm_start_internal_ratio={warm_start_ratio:.4f} | final_internal_ratio={ratio:.4f}"
    )
    warm_start_output_dtype_policy = output_dtype_policy
    if output_dtype_policy is None or output_dtype_policy != "runtime":
        warm_start_output_dtype_policy = "runtime"
        log("Using runtime dtype for warm-start student factors; final compression will use requested output dtype")
    whitening(
        model_name,
        model,
        offline_profile,
        warm_start_ratio,
        dev,
        init_scheme=init_scheme,
        output_dtype_policy=warm_start_output_dtype_policy,
    )
    if "opt" in model_name:
        layer_count = len(model.model.decoder.layers)
    else:
        layer_count = len(model.model.layers)
    on_policy_layer_count = math.ceil(layer_count * on_policy_layer_tail_ratio)
    on_policy_layer_start = layer_count - on_policy_layer_count
    hook_specs = [
        spec for spec in _build_low_rank_hook_specs(model_name, model)
        if spec[0] >= on_policy_layer_start
    ]
    log(
        f"Registered on-policy KD hook specs for {len(hook_specs)} low-rank projections | "
        f"tail_layer_ratio={on_policy_layer_tail_ratio:.4f} | "
        f"gradient_layers={on_policy_layer_count}/{layer_count} | "
        f"first_gradient_layer={on_policy_layer_start if on_policy_layer_count > 0 else 'none'}"
    )
    log(f"Moving active student model to {dev} for on-policy collection")
    model.to(dev)

    for round_idx in range(rounds):
        objective_is_on_policy = on_policy_loss == "reverse_kd"
        log(f"Start guided whitening round {round_idx + 1}/{rounds} | objective={on_policy_loss}")
        student_snapshot = _capture_module_snapshot(model, dense_paths)
        collector = _OnPolicyCovCollector(
            model,
            hook_specs,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            delta=alpha_delta,
            device=dev,
        )
        prompt_batches = build_prompt_batches(
            prompt_dataset,
            tokenizer,
            nsamples=prompt_nsamples,
            prompt_len=prompt_len,
            seqlen=model.seqlen,
            seed=seed + round_idx,
            batch_size=eval_batch_size,
        )
        log(
            f"Round {round_idx + 1}: prompt batches ready | batches={len(prompt_batches)} | "
            f"batch_size={eval_batch_size}"
        )

        try:
            running_loss = 0.0
            loss_label = f"{on_policy_loss}_loss"
            for batch_idx, prompts in enumerate(tqdm(prompt_batches, desc=f"collecting {on_policy_loss} covariances")):
                prompts = prompts.to(dev)
                model.zero_grad(set_to_none=True)
                if batch_idx == 0:
                    if objective_is_on_policy:
                        log(
                            f"Round {round_idx + 1}: sampling on-policy rollouts on {dev} | "
                            f"batch_size={prompts.shape[0]} | prompt_len={prompts.shape[1]} | "
                            f"rollout_len={rollout_len} | kv_cache={use_on_policy_kv_cache}"
                        )
                    else:
                        log(
                            f"Round {round_idx + 1}: using off-policy prompt-prefix objective on {dev} | "
                            f"batch_size={prompts.shape[0]} | prompt_len={prompts.shape[1]}"
                        )
                if objective_is_on_policy:
                    with torch.no_grad():
                        sequences = _sample_on_policy_sequences(
                            model,
                            prompts,
                            rollout_len=rollout_len,
                            generation_temperature=generation_temperature,
                            generation_top_p=generation_top_p,
                            use_kv_cache=use_on_policy_kv_cache,
                        )
                else:
                    sequences = prompts
                if batch_idx == 0:
                    log(
                        f"Round {round_idx + 1}: first objective batch ready | "
                        f"sequence_len={sequences.shape[1]}"
                    )
                collector.enabled = True
                student_logits = model(sequences, use_cache=False).logits
                teacher_logits = None
                if on_policy_loss in {"reverse_kd", "prefix_kd"}:
                    _apply_module_snapshot(model, dense_snapshot, device=dev, offload_replaced_to_cpu=False)
                    with torch.no_grad():
                        teacher_logits = model(sequences, use_cache=False).logits
                    _apply_module_snapshot(model, student_snapshot, device=dev, offload_replaced_to_cpu=False)
                loss, gradient_slice, loss_label = _compute_on_policy_objective(
                    on_policy_loss,
                    sequences,
                    student_logits,
                    teacher_logits,
                    prompt_len,
                    rollout_len,
                    kd_temperature=kd_temperature,
                )
                running_loss += loss.item()
                loss.backward()
                collector.consume_batch(gradient_slice)
                collector.enabled = False
                model.zero_grad(set_to_none=True)
                if batch_idx == 0 or batch_idx == len(prompt_batches) - 1:
                    log(
                        f"Round {round_idx + 1}: processed prompt batch {batch_idx + 1}/{len(prompt_batches)} | "
                        f"{loss_label}={loss.item():.6f} | sequence_len={sequences.shape[1]} | "
                        f"gradient_slice={gradient_slice}"
                    )
        finally:
            collector.enabled = False
            collector.close()
            _apply_module_snapshot(model, student_snapshot, device=dev, offload_replaced_to_cpu=False)

        active_stats = sum(1 for stat in collector.stats.values() if stat["count"] > 0)
        total_tokens = sum(stat["count"] for stat in collector.stats.values())
        mean_loss = running_loss / max(len(prompt_batches), 1)
        log(
            f"Round {round_idx + 1}: collected on-policy stats | active_projections={active_stats}/{len(collector.stats)} | "
            f"total_tokens={total_tokens} | mean_{loss_label}={mean_loss:.6f}"
        )
        gradient_profile = _build_gradient_whitening_profile(
            collector.stats,
            lambda0=lambda0,
            dev=dev,
            whitening_mode=gradient_whitening_mode,
            offline_profile=offline_profile,
        )
        _apply_module_snapshot(model, dense_snapshot, device=dev, offload_replaced_to_cpu=True)
        log(
            f"Round {round_idx + 1}: recompressing from dense teacher weights | "
            f"gradient tail layers={on_policy_layer_count}/{layer_count}; earlier layers use offline whitening"
        )
        gradient_whitening(
            model_name,
            model,
            gradient_profile,
            ratio,
            dev,
            init_scheme=init_scheme,
            offline_profile=offline_profile,
            gradient_layer_start=on_policy_layer_start,
            output_dtype_policy=output_dtype_policy,
        )
        model.to(dev)
        log(f"Round {round_idx + 1}: recompression complete")

    log("Finished on-policy reverse-KD guided whitening")
    return model



@torch.no_grad()
def profle_svdllm(name, model, calib_loader, dev):
    name = name.lower()
    if "llama" in name or "mistral" in name or "vicuna" in name or "qwen" in name:
        layers = model.model.layers
    elif "opt" in name:
        layers = model.model.decoder.layers
    model = model.to(dev)
    print("Start obtaining the whitening matrix...")
    def hook(module, input, output):
        inp = input[0].detach().float()
        if inp.dim() == 2:   # for opt
            inp = inp.unsqueeze(0)
        adds = torch.matmul(inp.transpose(1,2), inp)
        adds_sum = torch.sum(adds, dim=0)
        module.raw_scaling_diag_matrix += adds_sum
        del inp, adds, adds_sum
        torch.cuda.empty_cache()
    handles = []
    for name, module in model.named_modules():
        weight = _get_linear_like_weight(module)
        if weight is not None:
            module.raw_scaling_diag_matrix = 0
            handles.append(module.register_forward_hook(hook))
    for batch in tqdm(calib_loader):
        batch = {k: v.to(dev) for k, v in batch.items()}
        model(**batch)
    for handle in handles:
        handle.remove()
    torch.cuda.empty_cache()
    model = model.cpu()
    for i in range(len(layers)):
        subset = _find_linear_like_layers(layers[i])
        for name in subset:
            subset[name].raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix.cpu()
    profiling_mat = {}
    print("Start Cholesky Decomposition...")
    for i in tqdm(range(len(layers))):
        layer_profile = {}
        subset = _find_linear_like_layers(layers[i])
        for name in subset:
            raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix.to(dev)
            scaling_diag_matrix = _cholesky_with_jitter(raw_scaling_diag_matrix, dev, dtype=torch.float64)
            layer_profile[name] = scaling_diag_matrix.cpu()
            scaling_diag_matrix = raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix = None
            del scaling_diag_matrix, raw_scaling_diag_matrix, subset[name].raw_scaling_diag_matrix
            torch.cuda.empty_cache()
        profiling_mat[i] = layer_profile
    return profiling_mat
        

@torch.no_grad()
def profle_svdllm_low_resource(model_name, model, calib_loader, dev, profile_batch_size=8):
    model_name = model_name.lower()
    use_cache = model.config.use_cache
    model.config.use_cache = False
    is_device_mapped = _has_accelerate_device_map(model)
    input_dev = _module_device(model, dev) if is_device_mapped else torch.device(dev)
    if "opt" in model_name:
        layers = model.model.decoder.layers
        if not is_device_mapped:
            model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
            model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
            model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    else:
        layers = model.model.layers
        if not is_device_mapped:
            model.model.embed_tokens = model.model.embed_tokens.to(dev)
            model.model.norm = model.model.norm.to(dev)
    if not is_device_mapped:
        layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (len(calib_loader), model.seqlen, model.config.hidden_size), dtype=dtype, device="cpu"
    )
    cache = {'i': 0, 'layer_kwargs': []}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)
        def forward(self, inp, **kwargs):
            kwargs["use_cache"] = False
            inps[cache['i']] = inp.cpu()
            cache['layer_kwargs'].append(_move_nested_to_cpu(kwargs))
            cache['i'] += 1
            raise ValueError
    layers[0] = Catcher(layers[0])
    try:
        for batch in calib_loader:
            try:
                batch = {k: v.to(input_dev) for k, v in batch.items()}
                model(**batch, use_cache=False)
            except ValueError:
                pass
            finally:
                batch = None
                torch.cuda.empty_cache()
        layers[0] = layers[0].module
        if not is_device_mapped:
            layers[0] = layers[0].cpu()
        if "opt" in model_name:
            if not is_device_mapped:
                model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
                model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.cpu()
                model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        else:  
            if not is_device_mapped:
                model.model.embed_tokens = model.model.embed_tokens.cpu()
                model.model.norm = model.model.norm.cpu()
        torch.cuda.empty_cache()
        outs = torch.zeros_like(inps)
        profiling_mat = {}
        for i in tqdm(range(len(layers))):
            layer_profile = {}
            layer = layers[i]
            if not is_device_mapped:
                layer = layer.to(dev)
            layer_dev = _module_device(layer, dev)
            subset = _find_linear_like_layers(layer)
            def hook(module, input, output):
                inp = input[0].detach().float()
                if inp.dim() == 2:  # for opt
                    inp = inp.unsqueeze(0)
                adds = torch.matmul(inp.transpose(1,2), inp)
                adds_sum = torch.sum(adds, dim=0).cpu()
                module.scaling_diag_matrix += adds_sum
                del inp, adds, adds_sum, output
                torch.cuda.empty_cache()
            handles = []
            for name in subset:
                weight = _get_linear_like_weight(subset[name])
                if weight is None:
                    continue
                subset[name].scaling_diag_matrix = torch.zeros(
                    (weight.shape[1], weight.shape[1]),
                    dtype=torch.float32,
                    device="cpu",
                )
                handles.append(subset[name].register_forward_hook(hook))
            for start in range(0, inps.shape[0], profile_batch_size):
                end = min(start + profile_batch_size, inps.shape[0])
                layer_kwargs = _move_nested_to_device(_stack_nested(cache['layer_kwargs'][start:end]), layer_dev)
                layer_kwargs["use_cache"] = False
                layer_output = layer(inps[start:end].to(layer_dev), **layer_kwargs)
                outs[start:end] = layer_output[0].cpu()
                layer_output = None
                layer_kwargs = None
            for h in handles:
                h.remove()
            handles = None
            if not is_device_mapped:
                layer = layer.cpu()
            torch.cuda.empty_cache()
            for name in subset:
                if not hasattr(subset[name], "scaling_diag_matrix"):
                    continue
                raw_scaling_diag_matrix = subset[name].scaling_diag_matrix.to(layer_dev)
                scaling_diag_matrix = _cholesky_with_jitter(raw_scaling_diag_matrix, layer_dev, dtype=torch.float64)
                layer_profile[name] = scaling_diag_matrix.cpu()
                subset[name].scaling_diag_matrix = None
                scaling_diag_matrix = raw_scaling_diag_matrix = None
                del scaling_diag_matrix, raw_scaling_diag_matrix
                torch.cuda.empty_cache()
            if not is_device_mapped:
                layers[i] = layer.cpu()
            profiling_mat[i] = layer_profile
            inps = outs
            torch.cuda.empty_cache()
        return profiling_mat
    finally:
        model.config.use_cache = use_cache
     
 
def _assign_low_rank_weights(model_name, layer, name, svd_u, svd_v, svd_attn=None, svd_mlp=None, svd_decoder=None):
    if 'opt' in model_name:
        if "q_proj" in name:
            _ensure_factor_pair(svd_decoder.self_attn, "q_u_proj", "q_v_proj", svd_u, svd_v)
            svd_decoder.self_attn.q_u_proj.weight.data = svd_u
            svd_decoder.self_attn.q_v_proj.weight.data = svd_v
            svd_decoder.self_attn.q_u_proj.bias.data = layer.self_attn.q_proj.bias.data
        elif "k_proj" in name:
            _ensure_factor_pair(svd_decoder.self_attn, "k_u_proj", "k_v_proj", svd_u, svd_v)
            svd_decoder.self_attn.k_u_proj.weight.data = svd_u
            svd_decoder.self_attn.k_v_proj.weight.data = svd_v
            svd_decoder.self_attn.k_u_proj.bias.data = layer.self_attn.k_proj.bias.data
        elif "v_proj" in name:
            _ensure_factor_pair(svd_decoder.self_attn, "v_u_proj", "v_v_proj", svd_u, svd_v)
            svd_decoder.self_attn.v_u_proj.weight.data = svd_u
            svd_decoder.self_attn.v_v_proj.weight.data = svd_v
            svd_decoder.self_attn.v_u_proj.bias.data = layer.self_attn.v_proj.bias.data
        elif "out_proj" in name:
            _ensure_factor_pair(svd_decoder.self_attn, "out_u_proj", "out_v_proj", svd_u, svd_v)
            svd_decoder.self_attn.out_u_proj.weight.data = svd_u
            svd_decoder.self_attn.out_v_proj.weight.data = svd_v
            svd_decoder.self_attn.out_u_proj.bias.data = layer.self_attn.out_proj.bias.data
        elif "fc1" in name:
            _ensure_factor_pair(svd_decoder, "fc1_u_proj", "fc1_v_proj", svd_u, svd_v)
            svd_decoder.fc1_u_proj.weight.data = svd_u
            svd_decoder.fc1_v_proj.weight.data = svd_v
            svd_decoder.fc1_u_proj.bias.data = layer.fc1.bias.data
        elif "fc2" in name:
            _ensure_factor_pair(svd_decoder, "fc2_u_proj", "fc2_v_proj", svd_u, svd_v)
            svd_decoder.fc2_u_proj.weight.data = svd_u
            svd_decoder.fc2_v_proj.weight.data = svd_v
            svd_decoder.fc2_u_proj.bias.data = layer.fc2.bias.data
            svd_decoder.self_attn_layer_norm = layer.self_attn_layer_norm
            svd_decoder.final_layer_norm = layer.final_layer_norm
        return

    if "q_proj" in name:
        _ensure_factor_pair(svd_attn, "q_u_proj", "q_v_proj", svd_u, svd_v)
        svd_attn.q_u_proj.weight.data = svd_u
        svd_attn.q_v_proj.weight.data = svd_v
        if "qwen" in model_name and layer.self_attn.q_proj.bias is not None:
            svd_attn.q_u_proj.bias.data.copy_(layer.self_attn.q_proj.bias.data)
    elif "k_proj" in name:
        _ensure_factor_pair(svd_attn, "k_u_proj", "k_v_proj", svd_u, svd_v)
        svd_attn.k_u_proj.weight.data = svd_u
        svd_attn.k_v_proj.weight.data = svd_v
        if "qwen" in model_name and layer.self_attn.k_proj.bias is not None:
            svd_attn.k_u_proj.bias.data.copy_(layer.self_attn.k_proj.bias.data)
    elif "v_proj" in name:
        _ensure_factor_pair(svd_attn, "v_u_proj", "v_v_proj", svd_u, svd_v)
        svd_attn.v_u_proj.weight.data = svd_u
        svd_attn.v_v_proj.weight.data = svd_v
        if "qwen" in model_name and layer.self_attn.v_proj.bias is not None:
            svd_attn.v_u_proj.bias.data.copy_(layer.self_attn.v_proj.bias.data)
    elif "o_proj" in name:
        _ensure_factor_pair(svd_attn, "o_u_proj", "o_v_proj", svd_u, svd_v)
        svd_attn.o_u_proj.weight.data = svd_u
        svd_attn.o_v_proj.weight.data = svd_v
        if "qwen" in model_name and layer.self_attn.o_proj.bias is not None:
            svd_attn.o_u_proj.bias.data.copy_(layer.self_attn.o_proj.bias.data)
        if "qwen" in model_name:
            svd_attn.q_norm.weight.data.copy_(layer.self_attn.q_norm.weight.data)
            svd_attn.k_norm.weight.data.copy_(layer.self_attn.k_norm.weight.data)
        layer.self_attn = svd_attn
    elif "gate_proj" in name:
        _ensure_factor_pair(svd_mlp, "gate_u_proj", "gate_v_proj", svd_u, svd_v)
        svd_mlp.gate_u_proj.weight.data = svd_u
        svd_mlp.gate_v_proj.weight.data = svd_v
    elif "down_proj" in name:
        _ensure_factor_pair(svd_mlp, "down_u_proj", "down_v_proj", svd_u, svd_v)
        svd_mlp.down_u_proj.weight.data = svd_u
        svd_mlp.down_v_proj.weight.data = svd_v
    elif "up_proj" in name:
        _ensure_factor_pair(svd_mlp, "up_u_proj", "up_v_proj", svd_u, svd_v)
        svd_mlp.up_u_proj.weight.data = svd_u
        svd_mlp.up_v_proj.weight.data = svd_v
        layer.mlp = svd_mlp


@torch.no_grad()
def whitening(
    model_name,
    model,
    profiling_mat,
    ratio,
    dev,
    init_scheme="uniform",
    output_dtype_policy=None,
    dynamic_rank_allocation=False,
):
    model_name = model_name.lower()
    model.eval()
    if 'opt' in model_name:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers
    dynamic_rank_map = {}
    if dynamic_rank_allocation:
        dynamic_rank_map, _ = _dynamic_rank_allocation(model_name, model, profiling_mat, ratio, dev)
    print("Start SVD decomposition after whitening...")
    for i in tqdm(range(len(layers))):
        layer = layers[i]
        subset = _resolve_profiled_layers(layer, profiling_mat[i])
        #### Replace Attn, MLP ####
        if "llama" in model_name or "vicuna" in model_name:
            svd_attn = SVD_LlamaAttention(
                config=model.config,
                ratio=ratio,
                init_scheme=init_scheme,
                layer_idx=getattr(layer.self_attn, "layer_idx", None),
            )
            svd_mlp = SVD_LlamaMLP(hidden_size=layer.hidden_size, intermediate_size=model.config.intermediate_size, hidden_act=model.config.hidden_act, ratio=ratio, init_scheme=init_scheme)
        elif "mistral" in model_name:
            svd_attn = SVD_MistralAttention(
                config=model.config,
                ratio=ratio,
                init_scheme=init_scheme,
                layer_idx=getattr(layer.self_attn, "layer_idx", None),
            )
            svd_mlp = SVD_MistralMLP(config=model.config, ratio=ratio, init_scheme=init_scheme)
        elif "qwen" in model_name:
            svd_attn = SVD_Qwen3Attention(
                config=model.config,
                ratio=ratio,
                init_scheme=init_scheme,
                layer_idx=getattr(layer.self_attn, "layer_idx", None),
            )
            svd_mlp = SVD_Qwen3MLP(config=model.config, ratio=ratio, init_scheme=init_scheme)
        elif 'opt' in model_name:
            svd_decoder = SVDOPTDecoderLayer(model.config, ratio=ratio, init_scheme=init_scheme)
        #### Replace Attn, MLP ####
        for name in subset:
            scaling_diag_matrix = profiling_mat[i][name].to(dev)
            weight_tensor = _get_linear_like_weight(subset[name], expected_in_features=scaling_diag_matrix.shape[0])
            if weight_tensor is None:
                continue
            source_dtype = weight_tensor.dtype
            output_dtype = _resolve_output_dtype(source_dtype, output_dtype_policy)
            W = weight_tensor.detach().float().to(dev)
            try:
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            except Exception as e:
                print("Warning: scaling_diag_matrix is not full rank!")
                scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0]).to(dev)
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            scaling_diag_matrix = scaling_diag_matrix.float()
            scaling_matrix_inv = scaling_matrix_inv.float()
            W_scale = torch.matmul(W, scaling_diag_matrix)
            U, S, VT = torch.linalg.svd(W_scale, full_matrices=False)
            num_s_after_trunc = dynamic_rank_map.get((i, name), _low_rank_from_ratio(W.shape[0], W.shape[1], ratio))
            truc_s = S[:num_s_after_trunc]
            truc_u = U[:, :num_s_after_trunc]
            truc_v = torch.matmul(VT[:num_s_after_trunc, :], scaling_matrix_inv)
            truc_sigma = torch.diag(truc_s)
            #### Replace Attn, MLP ####
            sqrtSigma = torch.sqrt(truc_sigma)
            svd_u = torch.matmul(truc_u, sqrtSigma).cpu().to(output_dtype)
            svd_v = torch.matmul(sqrtSigma, truc_v).cpu().to(output_dtype)
            _assign_low_rank_weights(model_name, layer, name, svd_u, svd_v, svd_attn=svd_attn if 'opt' not in model_name else None, svd_mlp=svd_mlp if 'opt' not in model_name else None, svd_decoder=svd_decoder if 'opt' in model_name else None)
            W = W_scale = scaling_matrix_inv = scaling_diag_matrix = U = S = VT = truc_s = truc_u = truc_v = truc_sigma = sqrtSigma = svd_u = svd_v = None
            del W, W_scale, scaling_matrix_inv, scaling_diag_matrix, U, S, VT, truc_s, truc_u, truc_v, truc_sigma, sqrtSigma, svd_u, svd_v
            torch.cuda.empty_cache()
        if 'opt' in model_name:
            layers[i] = svd_decoder
        del layer
        torch.cuda.empty_cache()


@torch.no_grad()
def gradient_whitening(
    model_name,
    model,
    gradient_profile,
    ratio,
    dev,
    init_scheme="uniform",
    offline_profile=None,
    gradient_layer_start=0,
    output_dtype_policy=None,
):
    model_name = model_name.lower()
    model.eval()
    if 'opt' in model_name:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers
    print("Start hybrid gradient/offline-whitened SVD decomposition...")
    for i in tqdm(range(len(layers))):
        layer = layers[i]
        if i >= gradient_layer_start:
            layer_profile_keys = gradient_profile.get(i, {})
        else:
            layer_profile_keys = offline_profile[i] if offline_profile is not None else {}
        subset = _resolve_profiled_layers(layer, layer_profile_keys)
        if "llama" in model_name or "vicuna" in model_name:
            svd_attn = SVD_LlamaAttention(
                config=model.config,
                ratio=ratio,
                init_scheme=init_scheme,
                layer_idx=getattr(layer.self_attn, "layer_idx", None),
            )
            svd_mlp = SVD_LlamaMLP(hidden_size=layer.hidden_size, intermediate_size=model.config.intermediate_size, hidden_act=model.config.hidden_act, ratio=ratio, init_scheme=init_scheme)
        elif "mistral" in model_name:
            svd_attn = SVD_MistralAttention(
                config=model.config,
                ratio=ratio,
                init_scheme=init_scheme,
                layer_idx=getattr(layer.self_attn, "layer_idx", None),
            )
            svd_mlp = SVD_MistralMLP(config=model.config, ratio=ratio, init_scheme=init_scheme)
        elif "qwen" in model_name:
            svd_attn = SVD_Qwen3Attention(
                config=model.config,
                ratio=ratio,
                init_scheme=init_scheme,
                layer_idx=getattr(layer.self_attn, "layer_idx", None),
            )
            svd_mlp = SVD_Qwen3MLP(config=model.config, ratio=ratio, init_scheme=init_scheme)
        elif 'opt' in model_name:
            svd_decoder = SVDOPTDecoderLayer(model.config, ratio=ratio, init_scheme=init_scheme)

        for name in subset:
            expected_in = None
            if i >= gradient_layer_start:
                expected_in = gradient_profile[i][name]["x"].shape[0]
            elif offline_profile is not None:
                expected_in = offline_profile[i][name].shape[0]
            weight_tensor = _get_linear_like_weight(subset[name], expected_in_features=expected_in)
            if weight_tensor is None:
                continue
            source_dtype = weight_tensor.dtype
            output_dtype = _resolve_output_dtype(source_dtype, output_dtype_policy)
            W = weight_tensor.detach().float().to(dev)
            num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
            if i >= gradient_layer_start:
                factors = gradient_profile[i][name]
                input_factor = factors["x"].float().to(dev)
                grad_factor = factors["g"].float().to(dev)
                W_scale = grad_factor.transpose(0, 1).matmul(W).matmul(input_factor)
                U, S, VT = torch.linalg.svd(W_scale, full_matrices=False)
                truc_s = S[:num_s_after_trunc]
                sqrt_s = torch.sqrt(truc_s)
                left = U[:, :num_s_after_trunc] * sqrt_s.unsqueeze(0)
                right = sqrt_s.unsqueeze(1) * VT[:num_s_after_trunc, :]
                svd_u = torch.linalg.solve(
                    grad_factor.transpose(0, 1),
                    left,
                ).cpu().to(output_dtype)
                svd_v = torch.linalg.solve(
                    input_factor.transpose(0, 1),
                    right.transpose(0, 1),
                ).transpose(0, 1).cpu().to(output_dtype)
                del input_factor, grad_factor, left, right
            else:
                if offline_profile is None:
                    raise ValueError("offline_profile is required when gradient_layer_start excludes early layers")
                scaling_diag_matrix = offline_profile[i][name].to(dev)
                try:
                    scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
                except Exception:
                    scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0]).to(dev)
                    scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
                scaling_diag_matrix = scaling_diag_matrix.float()
                scaling_matrix_inv = scaling_matrix_inv.float()
                W_scale = torch.matmul(W, scaling_diag_matrix)
                U, S, VT = torch.linalg.svd(W_scale, full_matrices=False)
                truc_s = S[:num_s_after_trunc]
                sqrt_s = torch.sqrt(truc_s)
                svd_u = (U[:, :num_s_after_trunc] * sqrt_s.unsqueeze(0)).cpu().to(output_dtype)
                svd_v = (
                    sqrt_s.unsqueeze(1) * torch.matmul(VT[:num_s_after_trunc, :], scaling_matrix_inv)
                ).cpu().to(output_dtype)
                del scaling_diag_matrix, scaling_matrix_inv
            _assign_low_rank_weights(
                model_name,
                layer,
                name,
                svd_u,
                svd_v,
                svd_attn=svd_attn if 'opt' not in model_name else None,
                svd_mlp=svd_mlp if 'opt' not in model_name else None,
                svd_decoder=svd_decoder if 'opt' in model_name else None,
            )
            del W, W_scale, U, S, VT, truc_s, sqrt_s, svd_u, svd_v
            torch.cuda.empty_cache()
        if 'opt' in model_name:
            layers[i] = svd_decoder
        del layer
        torch.cuda.empty_cache()


@torch.no_grad()
def whitening_local_update(
    model_name,
    model,
    dataloader,
    profiling_mat,
    ratio,
    dev,
    direct_update=False,
    init_scheme="uniform",
    dynamic_rank_allocation=False,
):
    model_name = model_name.lower()
    print("Start SVD decomposition then update...")
    use_cache = model.config.use_cache
    model.config.use_cache = False
    if "opt" in model_name:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    else:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (len(dataloader), model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'layer_kwargs': []}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['layer_kwargs'].append(kwargs)
            cache['i'] += 1
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    dynamic_rank_map = {}
    if dynamic_rank_allocation and profiling_mat is not None and not direct_update:
        dynamic_rank_map, _ = _dynamic_rank_allocation(model_name, model, profiling_mat, ratio, dev)
    for i in tqdm(range(len(layers))):
        layer = layers[i].to(dev)
        subset = find_layers(layer)
        gpts = {}
        if "llama" in model_name or "vicuna" in model_name:
            svd_attn = SVD_LlamaAttention(
                config=model.config,
                ratio=ratio,
                init_scheme=init_scheme,
                layer_idx=getattr(layer.self_attn, "layer_idx", None),
            )
            svd_mlp = SVD_LlamaMLP(hidden_size=layer.hidden_size, intermediate_size=model.config.intermediate_size, hidden_act=model.config.hidden_act, ratio=ratio, init_scheme=init_scheme)
        elif "mistral" in model_name:
            svd_attn = SVD_MistralAttention(
                config=model.config,
                ratio=ratio,
                init_scheme=init_scheme,
                layer_idx=getattr(layer.self_attn, "layer_idx", None),
            )
            svd_mlp = SVD_MistralMLP(config=model.config, ratio=ratio, init_scheme=init_scheme)
        elif "qwen" in model_name:
            svd_attn = SVD_Qwen3Attention(
                config=model.config,
                ratio=ratio,
                init_scheme=init_scheme,
                layer_idx=getattr(layer.self_attn, "layer_idx", None),
            )
            svd_mlp = SVD_Qwen3MLP(config=model.config, ratio=ratio, init_scheme=init_scheme)
        elif 'opt' in model_name:
            svd_decoder = SVDOPTDecoderLayer(model.config, ratio=ratio, init_scheme=init_scheme)
        for name in subset:
            if profiling_mat is not None:
                scaling_diag_matrix = profiling_mat[i][name].to(dev)
            else: 
                scaling_diag_matrix = None
            gpts[name] = local_update(
                subset[name],
                scaling_diag_matrix=scaling_diag_matrix,
                ratio=ratio,
                name=name,
                direct_update=direct_update,
                rank=dynamic_rank_map.get((i, name)),
            )
        
        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch_update_u(inp[0].data, out.data)
            return tmp
        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(inps.shape[0]):
            outs[j] = layer(inps[j].unsqueeze(0), **cache['layer_kwargs'][j])[0]
        for h in handles:
            h.remove()
        for name in gpts:
            svd_u, svd_v = gpts[name].fasterprune()
            svd_u, svd_v = svd_u.to(dtype), svd_v.to(dtype)
            _assign_low_rank_weights(model_name, layer, name, svd_u, svd_v, svd_attn=svd_attn if 'opt' not in model_name else None, svd_mlp=svd_mlp if 'opt' not in model_name else None, svd_decoder=svd_decoder if 'opt' in model_name else None)
        layer = layer.to(dev)
        for j in range(inps.shape[0]):
            outs[j] = layer(inps[j].unsqueeze(0), **cache['layer_kwargs'][j])[0]
        layers[i] = layer.cpu()
        del gpts
        torch.cuda.empty_cache()
        inps = outs
        outs = None
        del outs
    model.config.use_cache = use_cache


class local_update:
    def __init__(self, layer, scaling_diag_matrix, ratio, name, direct_update=False, rank=None):
        self.layer = layer
        self.name = name
        self.dev = self.layer.weight.device
        # W = layer.weight.data.clone()
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        if direct_update:
            self.U, self.S, self.VT = torch.linalg.svd(W.data, full_matrices=False)
        else: 
            try:
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            except Exception as e:
                print("Warning: scaling_diag_matrix is not full rank!")
                scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0])
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            scaling_diag_matrix = scaling_diag_matrix.float()
            scaling_matrix_inv = scaling_matrix_inv.float()
            W_scale = torch.matmul(W, scaling_diag_matrix)
            self.U, self.S, self.VT = torch.linalg.svd(W_scale, full_matrices=False)  
        # trucation SVD
        num_s_after_trunc = rank if rank is not None else _low_rank_from_ratio(W.shape[0], W.shape[1], ratio)
        self.truc_s = self.S[:num_s_after_trunc].cuda()
        self.truc_u = self.U[:, :num_s_after_trunc].cuda()
        if direct_update:
            self.truc_v = self.VT[:num_s_after_trunc, :].cuda()
        else:
            self.truc_v = torch.matmul(self.VT[:num_s_after_trunc, :].cuda(), scaling_matrix_inv)
        self.truc_sigma = torch.diag(self.truc_s)
        self.new_w = torch.matmul(self.truc_u, torch.matmul(self.truc_sigma, self.truc_v[:num_s_after_trunc, :]))
        # intialize H for close form solution
        self.updated_err = self.error = 0

    def add_batch_update_u(self, inp, out):
        inps = inp.view(inp.shape[0] * inp.shape[1], inp.shape[2])
        outs = out.view(out.shape[0] * out.shape[1], out.shape[2])
        new_w = torch.matmul(self.truc_u, torch.matmul(self.truc_sigma, self.truc_v))
        new_output = inps.matmul(new_w.t())
        self.error = torch.sqrt(torch.sum((outs - new_output)**2)).item() / torch.norm(outs, p='fro').item()
        # print(f"truncted error: {self.error}")
        x =  torch.matmul(torch.matmul(inps, self.truc_v.T), self.truc_sigma)
        self.updated_uT = torch.linalg.lstsq(x,outs).solution
        updated_output = torch.matmul(torch.matmul(torch.matmul(inps, self.truc_v.T), self.truc_sigma), self.updated_uT)
        self.updated_error = torch.sqrt(torch.sum((outs - updated_output)**2)).item() / torch.norm(outs, p='fro').item()
        # print(f"updated error: {self.updated_error}")
        inps = outs = new_output = updated_output = x = new_w = None
        del inps, outs, new_output, updated_output, x, new_w
        torch.cuda.empty_cache()
        # print(f"Finish {self.name}"
    
    def fasterprune(self):
        sqrtSigma = torch.sqrt(self.truc_sigma)
        self.appendU = self.updated_uT.t().matmul(sqrtSigma)
        self.appendV = sqrtSigma.matmul(self.truc_v)
        return self.appendU, self.appendV


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='jeffwan/llama-7b-hf', help='LLaMA model to load, pass `jeffwan/llama-7b-hf`')
    parser.add_argument('--model_path', type=str, default=None, help='local compressed model path or whitening information path')
    parser.add_argument('--ratio', type=float, default=0.2, help='Target compression ratio,(0,1), default=0.2, means only keeping about 20% of the params.')
    parser.add_argument('--run_low_resource', action='store_true', help='whether to run whitening in low resource, exp, compress LLaMA-7B below 15G gpu')
    parser.add_argument(
        '--dataset',
        type=str,
        default='wikitext2',
        help='Calibration data source. Supports single datasets like [wikitext2, ptb, c4, evol-codealpaca, theblackcat102/evol-codealpaca-v1, tulu-math, allenai/tulu-3-sft-personas-math] and mixtures like mix:wikitext2,evol-codealpaca,tulu-math',
    )
    parser.add_argument('--whitening_nsamples', type=int, default=256, help='Number of calibration data samples for whitening.')
    parser.add_argument('--updating_nsamples', type=int, default=16, help='Number of calibration data samples for udpating.')
    parser.add_argument('--save_path', type=str, default=None, help='the path to save the compressed model checkpoints.`')
    parser.add_argument('--profiling_mat_path', type=str, default=None, help='Local path to load the profiling matrices`')
    parser.add_argument('--seed',type=int, default=0, help='Seed for sampling the calibration data')
    parser.add_argument('--DEV', type=str, default="cuda", help='device')
    parser.add_argument('--model_seq_len', type=int, default=2048, help='the default sequence length of the LLM')
    parser.add_argument(
        '--profile_batch_size',
        type=int,
        default=8,
        help='Mini-batch size used inside low-resource whitening profiling. Larger is faster but uses more GPU memory.',
    )
    parser.add_argument('--eval_batch_size', type=int, default=4, help='inference bactch size')
    parser.add_argument('--gen_seq_len', type=int, default=1024, help='generated sequence len for efficiency evaluation')
    parser.add_argument('--step', type=int, default=4, help='the step to run the compression')
    parser.add_argument('--lora', type=str, default=None, help='the lora updated weight path to run the accuracy evaluation')
    parser.add_argument('--offline_dataset', type=str, default='c4', help='Offline covariance dataset for step 6.')
    parser.add_argument(
        '--warm_start_ratio',
        type=float,
        default=None,
        help='Optional step 6 warm-start compression ratio, using the same convention as --ratio. If omitted, warm start uses --ratio.',
    )
    parser.add_argument('--on_policy_dataset', type=str, default='mix:evol-codealpaca,tulu-math', help='Prompt dataset used for on-policy reverse KD in step 6.')
    parser.add_argument('--on_policy_prompt_len', type=int, default=128, help='Prompt length for on-policy reverse KD in step 6.')
    parser.add_argument('--on_policy_rollout_len', type=int, default=64, help='Generated continuation length for on-policy reverse KD in step 6.')
    parser.add_argument('--on_policy_prompt_nsamples', type=int, default=16, help='Number of prompt samples used for on-policy reverse KD in step 6.')
    parser.add_argument('--on_policy_rounds', type=int, default=1, help='Number of reverse-KD-guided whitening rounds in step 6.')
    parser.add_argument('--kd_temperature', type=float, default=2.0, help='Distillation temperature tau for reverse KD in step 6.')
    parser.add_argument(
        '--on_policy_loss',
        type=str,
        default='reverse_kd',
        choices=['reverse_kd', 'lm', 'prefix_kd'],
        help='Step 6 gradient objective: reverse_kd on generated rollout tokens, lm cross-entropy on prompt-prefix tokens, or prefix_kd on prompt-prefix logits.',
    )
    parser.add_argument('--generation_temperature', type=float, default=0.7, help='Sampling temperature for student rollouts in step 6.')
    parser.add_argument('--generation_top_p', type=float, default=0.9, help='Top-p sampling threshold for student rollouts in step 6.')
    parser.add_argument(
        '--disable_on_policy_kv_cache',
        action='store_true',
        help='Disable KV-cache rollout generation in step 6 and use full-prefix recomputation.',
    )
    parser.add_argument('--cov_rho', type=float, default=0.3, help='Mixing weight rho for on-policy covariance in step 6.')
    parser.add_argument('--alpha_min', type=float, default=0.1, help='Minimum token importance clamp for step 6.')
    parser.add_argument('--alpha_max', type=float, default=10.0, help='Maximum token importance clamp for step 6.')
    parser.add_argument('--alpha_delta', type=float, default=1e-6, help='Normalization epsilon for token importance in step 6.')
    parser.add_argument('--lambda0', type=float, default=1e-6, help='Diagonal stabilization scale for mixed covariances in step 6.')
    parser.add_argument(
        '--gradient_whitening_mode',
        type=str,
        default='both',
        choices=['both', 'grad_only'],
        help='Step 6 whitening metric: both uses input and output-gradient covariances; grad_only uses output-gradient covariance with identity input metric.',
    )
    parser.add_argument(
        '--on_policy_layer_tail_ratio',
        type=float,
        default=1.0,
        help='Fraction of final transformer layers that use on-policy gradient whitening in step 6. Earlier layers use the original offline whitening profile.',
    )
    parser.add_argument(
        '--init_scheme',
        type=str,
        default='uniform',
        choices=['uniform', 'xavier_uniform', 'kaiming_uniform', 'default'],
        help='Initialization scheme for freshly created SVD low-rank modules before decomposition weights are loaded.',
    )
    parser.add_argument(
        '--svd_output_dtype',
        type=str,
        default='original',
        choices=['original', 'runtime', 'float8_e4m3fn', 'float8_e5m2', 'float16', 'bfloat16', 'float32'],
        help=(
            'Dtype for saved SVD factor weights. original preserves source dtype, including FP8; '
            'runtime converts FP8 source weights to fp16 for executable low-rank modules.'
        ),
    )
    parser.add_argument(
        '--dynamic_rank_allocation',
        action='store_true',
        help='Apply SVD-LLM V2 layer-sensitivity based dynamic rank allocation during whitening.',
    )
    
    args = parser.parse_args()
    user_ratio = args.ratio
    user_warm_start_ratio = args.warm_start_ratio
    args.ratio = 1- args.ratio
    if args.warm_start_ratio is not None:
        args.warm_start_ratio = 1 - args.warm_start_ratio
    if not (0.0 <= args.on_policy_layer_tail_ratio <= 1.0):
        raise ValueError("--on_policy_layer_tail_ratio must be in [0, 1]")
    svd_output_dtype_policy = _parse_svd_output_dtype(args.svd_output_dtype)
    if args.step == 1:
        model, tokenizer = get_model_from_huggingface(model_id=args.model)
        model = model.eval()
        if args.profiling_mat_path is None:
            cali_white_data = get_calib_train_data(args.dataset, tokenizer, args.whitening_nsamples, seqlen=args.model_seq_len)
            profiling_mat = profle_svdllm_low_resource(
                args.model, model, cali_white_data, args.DEV, profile_batch_size=args.profile_batch_size
            )
            if args.save_path is not None:
                torch.save(profiling_mat, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") + '_profiling_'+ args.dataset + '_' + str(args.whitening_nsamples)  + '_' + str(args.seed)+ '.pt')
        else:
            profiling_mat = torch.load(args.profiling_mat_path)
        whitening(
            args.model,
            model,
            profiling_mat,
            args.ratio,
            args.DEV,
            init_scheme=args.init_scheme,
            output_dtype_policy=svd_output_dtype_policy,
            dynamic_rank_allocation=args.dynamic_rank_allocation,
        )
        if args.save_path is not None:
            save_model_checkpoint(model, tokenizer, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") +'_whitening_only_code_math_wiki_nolast' + str(args.ratio) + '.pt')   # fp32
    elif args.step == 2:
        model, tokenizer = get_model_from_huggingface(model_id=args.model)
        dataloader, _ = get_loaders(args.dataset, nsamples=args.updating_nsamples, seed=args.seed, tokenizer=tokenizer, seqlen=args.model_seq_len)
        model = model.eval()
        model = model.float()  # need to set to float
        if args.profiling_mat_path is None:
            cali_white_data = get_calib_train_data(args.dataset, tokenizer, args.whitening_nsamples, seqlen=args.model_seq_len)
            profiling_mat = profle_svdllm_low_resource(
                args.model, model, cali_white_data, args.DEV, profile_batch_size=args.profile_batch_size
            )
            if args.save_path is not None:
                torch.save(profiling_mat, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") + '_profiling_'+ args.dataset + '_' + str(args.whitening_nsamples)  + '_' + str(args.seed)+ '.pt')
        else:
            profiling_mat = torch.load(args.profiling_mat_path)
        whitening_local_update(
            args.model,
            model,
            dataloader,
            profiling_mat,
            args.ratio,
            args.DEV,
            init_scheme=args.init_scheme,
            dynamic_rank_allocation=args.dynamic_rank_allocation,
        )
        if args.save_path is not None:
            save_model_checkpoint(model, tokenizer, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") +'_whitening_then_update_' + str(args.ratio) + '.pt')  # fp32
    elif args.step == 3:
        model, tokenizer = get_model_from_huggingface(args.model)
        model = model.eval()
        model = model.float()
        dataloader, _ = get_loaders(args.dataset, nsamples=args.updating_nsamples, seed=args.seed, tokenizer=tokenizer, seqlen=args.model_seq_len)
        whitening_local_update(model_name=args.model, model=model, dataloader=dataloader, profiling_mat=None, ratio=args.ratio, dev=args.DEV, direct_update=True, init_scheme=args.init_scheme)
        if args.save_path is not None:
            save_model_checkpoint(model, tokenizer, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") +'_update_only_' + str(args.ratio) + '.pt')   # fp32
    elif args.step == 6:
        warm_start_log = (
            f"{user_warm_start_ratio:.4f}"
            if user_warm_start_ratio is not None
            else f"{user_ratio:.4f} (same as --ratio)"
        )
        log(
            f"Step 6 selected: on-policy reverse-KD guided whitening | model={args.model} | "
            f"final_ratio_arg={user_ratio:.4f} | warm_start_ratio_arg={warm_start_log} | device={args.DEV}"
        )
        log(
            f"Offline calibration: dataset={args.offline_dataset} | nsamples={args.whitening_nsamples} | "
            f"seq_len={args.model_seq_len}"
        )
        log(
            f"On-policy KD prompts: dataset={args.on_policy_dataset} | prompt_nsamples={args.on_policy_prompt_nsamples} | "
            f"prompt_len={args.on_policy_prompt_len} | rollout_len={args.on_policy_rollout_len} | "
            f"gradient_whitening_mode={args.gradient_whitening_mode} | "
            f"loss={args.on_policy_loss} | "
            f"kv_cache={not args.disable_on_policy_kv_cache if args.on_policy_loss == 'reverse_kd' else 'unused'} | "
            f"tail_layer_ratio={args.on_policy_layer_tail_ratio}"
        )
        model, tokenizer = get_model_from_huggingface(model_id=args.model)
        model = model.eval()
        if args.profiling_mat_path is None:
            log("No offline profiling matrix provided; collecting offline covariance profile now")
            c4_white_data = get_calib_train_data(
                args.offline_dataset,
                tokenizer,
                args.whitening_nsamples,
                seqlen=args.model_seq_len,
                seed=args.seed,
            )
            offline_profile = profle_svdllm_low_resource(
                args.model, model, c4_white_data, args.DEV, profile_batch_size=args.profile_batch_size
            )
            log("Offline covariance profile collection complete")
            if args.save_path is not None:
                torch.save(offline_profile, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") + '_profiling_'+ args.dataset + '_' + str(args.whitening_nsamples)  + '_' + str(args.seed)+ '.pt')
        else:
            log(f"Loading offline profiling matrix from {args.profiling_mat_path}")
            offline_profile = torch.load(args.profiling_mat_path)
        on_policy_reverse_kd_guided_whitening(
            model_name=args.model,
            model=model,
            tokenizer=tokenizer,
            ratio=args.ratio,
            warm_start_ratio=args.warm_start_ratio,
            dev=args.DEV,
            offline_profile=offline_profile,
            prompt_dataset=args.on_policy_dataset,
            prompt_nsamples=args.on_policy_prompt_nsamples,
            prompt_len=args.on_policy_prompt_len,
            rollout_len=args.on_policy_rollout_len,
            kd_temperature=args.kd_temperature,
            generation_temperature=args.generation_temperature,
            generation_top_p=args.generation_top_p,
            rho=args.cov_rho,
            alpha_min=args.alpha_min,
            alpha_max=args.alpha_max,
            alpha_delta=args.alpha_delta,
            lambda0=args.lambda0,
            rounds=args.on_policy_rounds,
            eval_batch_size=args.eval_batch_size,
            seed=args.seed,
            init_scheme=args.init_scheme,
            gradient_whitening_mode=args.gradient_whitening_mode,
            use_on_policy_kv_cache=not args.disable_on_policy_kv_cache,
            on_policy_layer_tail_ratio=args.on_policy_layer_tail_ratio,
            output_dtype_policy=svd_output_dtype_policy,
            on_policy_loss=args.on_policy_loss,
        )
        if args.save_path is not None:
            output_path = (
                args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") + f'_on_policy_{args.on_policy_loss}' + 'comp_ratio' + str(args.ratio) + f'wp_ratio_{args.warm_start_ratio}' + f'lt_ratio_{args.on_policy_layer_tail_ratio}' +  args.offline_dataset + '_' + str(args.whitening_nsamples) + str(args.on_policy_prompt_nsamples) + str(args.gradient_whitening_mode) + '.pt'
            )
            log(f"Saving step 6 checkpoint to {output_path}")
            save_model_checkpoint(model, tokenizer, output_path)
    elif args.step >= 4:
        print(f"evaluating {args.model_path}...")
        if args.model_path == "original":
            model, tokenizer = get_model_from_huggingface(args.model)
        else:
            model, tokenizer = get_model_from_local(args.model_path)
            if args.lora is not None:
                from utils.peft import PeftModel
                model = PeftModel.from_pretrained(
                    model,
                    args.lora,
                    torch_dtype=torch.float16,
                )
                model = model.merge_and_unload()
                save_model_checkpoint(model, tokenizer, args.lora + '/merge.pt')
        model.eval()
        model = model.half()
        model = model.to(args.DEV)
        if args.step == 4:
            ppl_eval(model, tokenizer, datasets=['wikitext2', 'ptb', 'c4'], model_seq_len=args.model_seq_len, batch_size=args.eval_batch_size, device=args.DEV)
        elif args.step == 5:
            eff_eval(model, tokenizer, generated_len=args.gen_seq_len, batch_size=args.eval_batch_size, device=args.DEV)
