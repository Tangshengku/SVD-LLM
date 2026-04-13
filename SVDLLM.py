#coding:utf8
import os
import sys
import argparse
from types import SimpleNamespace
import torch.jit
from tqdm import tqdm
import torch
import torch.nn as nn

from utils.data_utils import *
from component.svd_llama import SVD_LlamaAttention, SVD_LlamaMLP
from component.svd_mistral import SVD_MistralAttention, SVD_MistralMLP
from component.svd_opt import SVDOPTDecoderLayer
from utils.model_utils import *
from evaluater import * 

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)


def _model_name(model_name):
    return model_name.lower()


def is_opt_model(model_name):
    return "opt" in _model_name(model_name)


def is_llama_family(model_name):
    name = _model_name(model_name)
    return "llama" in name or "vicuna" in name


def is_mistral_family(model_name):
    name = _model_name(model_name)
    return "mistral" in name or "qwen" in name


def is_llama_style_model(model_name):
    return is_llama_family(model_name) or is_mistral_family(model_name)


def resolve_runtime_device(dev):
    if isinstance(dev, str) and dev.lower() == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        return "cpu"
    return dev


def is_sharded_model(model):
    return hasattr(model, "hf_device_map") and getattr(model, "hf_device_map", None) not in [None, {}]


def ensure_unsharded_model(model, model_name, dev):
    return dev


def _to_cpu_structure(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, tuple):
        return tuple(_to_cpu_structure(x) for x in obj)
    if isinstance(obj, list):
        return [_to_cpu_structure(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_cpu_structure(v) for k, v in obj.items()}
    return obj


def _to_device_structure(obj, dev):
    if torch.is_tensor(obj):
        return obj.to(dev)
    if isinstance(obj, tuple):
        return tuple(_to_device_structure(x, dev) for x in obj)
    if isinstance(obj, list):
        return [_to_device_structure(x, dev) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_device_structure(v, dev) for k, v in obj.items()}
    return obj


def _infer_module_device(module, fallback_dev):
    for param in module.parameters(recurse=False):
        return str(param.device)
    for buf in module.buffers(recurse=False):
        return str(buf.device)
    for param in module.parameters():
        return str(param.device)
    for buf in module.buffers():
        return str(buf.device)
    return fallback_dev


def _run_layer_with_kwargs(layer, hidden_states, layer_kwargs, dev):
    layer_dev = _infer_module_device(layer, dev)
    kwargs = _to_device_structure(layer_kwargs, layer_dev)
    hidden_states = hidden_states.to(layer_dev)
    return layer(hidden_states, **kwargs)[0]


def unwrap_model_for_save(module):
    for child in module.modules():
        if hasattr(child, "_old_forward"):
            child.forward = child._old_forward
            delattr(child, "_old_forward")
        if hasattr(child, "_hf_hook"):
            delattr(child, "_hf_hook")
    return module


def _infer_input_device(model, fallback_dev):
    candidate_modules = []
    if hasattr(model, "model"):
        if hasattr(model.model, "embed_tokens"):
            candidate_modules.append(model.model.embed_tokens)
        if hasattr(model.model, "decoder") and hasattr(model.model.decoder, "embed_tokens"):
            candidate_modules.append(model.model.decoder.embed_tokens)
    for module in candidate_modules:
        return _infer_module_device(module, fallback_dev)
    return fallback_dev


def _prepare_model_inputs(batch, model, dev):
    target_dev = _infer_input_device(model, dev) if is_sharded_model(model) else dev
    if isinstance(batch, dict):
        return {k: v.to(target_dev) if torch.is_tensor(v) else v for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return [v.to(target_dev) if torch.is_tensor(v) else v for v in batch]
    return batch.to(target_dev) if torch.is_tensor(batch) else batch



@torch.no_grad()
def profle_svdllm(name, model, calib_loader, dev):
    if is_llama_style_model(name):
        layers = model.model.layers
    elif is_opt_model(name):
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
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.raw_scaling_diag_matrix = 0
            module.register_forward_hook(hook)
    for batch in tqdm(calib_loader):
        batch = _prepare_model_inputs(batch, model, dev)
        model(**batch)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module._forward_hooks.clear()
    torch.cuda.empty_cache()
    model = model.cpu()
    for i in range(len(layers)):
        subset = find_layers(layers[i])
        for name in subset:
            subset[name].raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix.cpu()
    profiling_mat = {}
    print("Start Cholesky Decomposition...")
    for i in tqdm(range(len(layers))):
        layer_profile = {}
        subset = find_layers(layers[i])
        for name in subset:
            raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix.double().to(dev)
            try:
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
            except Exception as e:
                print("Warning: eigen scaling_diag_matrix is not positive!")
                eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
                raw_scaling_diag_matrix += (- eigenvalues[0] + 1e-6) * torch.eye(raw_scaling_diag_matrix.shape[0]).to(dev)
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
                eigenvalues = None
                del eigenvalues
            layer_profile[name] = scaling_diag_matrix.cpu()
            scaling_diag_matrix = raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix = None
            del scaling_diag_matrix, raw_scaling_diag_matrix, subset[name].raw_scaling_diag_matrix
            torch.cuda.empty_cache()
        profiling_mat[i] = layer_profile
    return profiling_mat
        

@torch.no_grad()
def profle_svdllm_low_resource(model_name, model, calib_loader, dev):
    sharded_model = is_sharded_model(model)
    if is_opt_model(model_name):
        layers = model.model.decoder.layers
        if not sharded_model:
            model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
            model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
            model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    else:
        layers = model.model.layers
        if not sharded_model:
            model.model.embed_tokens = model.model.embed_tokens.to(dev)
            model.model.norm = model.model.norm.to(dev)
    if not sharded_model:
        layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (len(calib_loader), model.seqlen, model.config.hidden_size), dtype=dtype
    )
    cache = {'i': 0, 'layer_kwargs': []}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp.cpu()
            cache['layer_kwargs'].append(_to_cpu_structure(kwargs))
            cache['i'] += 1
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in calib_loader:
        try:
            batch = _prepare_model_inputs(batch, model, dev)
            model(**batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    if not sharded_model:
        layers[0] = layers[0].cpu()
    if is_opt_model(model_name):
        if not sharded_model:
            model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
            model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.cpu()
            model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    else:  
        if not sharded_model:
            model.model.embed_tokens = model.model.embed_tokens.cpu()
            model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    layer_kwargs = cache['layer_kwargs']
    profiling_mat = {}
    for i in tqdm(range(len(layers))):
        layer_profile = {}
        layer = layers[i] if sharded_model else layers[i].to(dev)
        subset = find_layers(layer)        
        def hook(module, input, output):
            inp = input[0].detach().float()
            if inp.dim() == 2:  # for opt
                inp = inp.unsqueeze(0)
            adds = torch.matmul(inp.transpose(1,2), inp)
            adds_sum = torch.sum(adds, dim=0)
            module.scaling_diag_matrix += adds_sum
            del inp, adds, adds_sum, output
            torch.cuda.empty_cache()
        handles = []
        for name in subset:
            subset[name].scaling_diag_matrix = 0
            handles.append(subset[name].register_forward_hook(hook))
        for j in range(inps.shape[0]):
            outs[j] = _run_layer_with_kwargs(layer, inps[j].unsqueeze(0), layer_kwargs[j], dev)
        for h in handles:
            h.remove()
        if not sharded_model:
            layer = layer.cpu()
        for name in subset:
            subset[name].scaling_diag_matrix = subset[name].scaling_diag_matrix.cpu()
        torch.cuda.empty_cache()
        for name in subset:
            raw_scaling_diag_matrix = subset[name].scaling_diag_matrix.double().to(dev)
            try:
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
            except Exception as e:
                print("Warning: eigen scaling_diag_matrix is not positive!")
                eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
                raw_scaling_diag_matrix += (- eigenvalues[0] + 1e-6) * torch.eye(raw_scaling_diag_matrix.shape[0]).to(dev)
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
                eigenvalues = None
                del eigenvalues
            layer_profile[name] = scaling_diag_matrix.cpu()
            scaling_diag_matrix = raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix = None
            del scaling_diag_matrix, raw_scaling_diag_matrix, subset[name].raw_scaling_diag_matrix
            torch.cuda.empty_cache()
        if not sharded_model:
            layers[i] = layer.cpu()
        profiling_mat[i] = layer_profile
        inps = outs.cpu()
        layer_kwargs = [_to_cpu_structure({k: v for k, v in kwargs.items()}) for kwargs in layer_kwargs]
        torch.cuda.empty_cache()
    return profiling_mat
     
 
@torch.no_grad()
def whitening(model_name, model, profiling_mat, ratio, dev):
    model.eval()
    if is_opt_model(model_name):
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers
    print("Start SVD decomposition after whitening...")
    for i in tqdm(range(len(layers))):
        layer = layers[i]
        subset = find_layers(layer)
        #### Replace Attn, MLP ####
        if is_llama_family(model_name):
            svd_attn = SVD_LlamaAttention(config=model.config, ratio=ratio)
            svd_mlp = SVD_LlamaMLP(hidden_size=layer.hidden_size, intermediate_size=model.config.intermediate_size, hidden_act=model.config.hidden_act, ratio=ratio)
        elif is_mistral_family(model_name):
            svd_attn = SVD_MistralAttention(config=model.config, ratio=ratio)
            svd_mlp = SVD_MistralMLP(config=model.config, ratio=ratio)
        elif is_opt_model(model_name):
            svd_decoder = SVDOPTDecoderLayer(model.config, ratio=ratio)
        #### Replace Attn, MLP ####
        for name in subset:
            W = subset[name].weight.data.float().to(dev)
            dtype = W.dtype
            scaling_diag_matrix = profiling_mat[i][name].to(dev)
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
            num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
            truc_s = S[:num_s_after_trunc]
            truc_u = U[:, :num_s_after_trunc]
            truc_v = torch.matmul(VT[:num_s_after_trunc, :], scaling_matrix_inv)
            truc_sigma = torch.diag(truc_s)
            #### Replace Attn, MLP ####
            sqrtSigma = torch.sqrt(truc_sigma)
            svd_u = torch.matmul(truc_u, sqrtSigma).cpu().to(dtype)
            svd_v = torch.matmul(sqrtSigma, truc_v).cpu().to(dtype)
            if is_opt_model(model_name):
                if "q_proj" in name:
                    svd_decoder.self_attn.q_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.q_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.q_u_proj.bias.data = layer.self_attn.q_proj.bias.data  # the linear layer in OPT has bias, which is different from LLaMA and Mistral
                elif "k_proj" in name:
                    svd_decoder.self_attn.k_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.k_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.k_u_proj.bias.data = layer.self_attn.k_proj.bias.data
                elif "v_proj" in name:
                    svd_decoder.self_attn.v_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.v_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.v_u_proj.bias.data = layer.self_attn.v_proj.bias.data
                elif "out_proj" in name:
                    svd_decoder.self_attn.out_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.out_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.out_u_proj.bias.data = layer.self_attn.out_proj.bias.data
                elif "fc1" in name:
                    svd_decoder.fc1_u_proj.weight.data = svd_u
                    svd_decoder.fc1_v_proj.weight.data = svd_v
                    svd_decoder.fc1_u_proj.bias.data = layer.fc1.bias.data
                elif "fc2" in name:
                    svd_decoder.fc2_u_proj.weight.data = svd_u
                    svd_decoder.fc2_v_proj.weight.data = svd_v
                    svd_decoder.fc2_u_proj.bias.data = layer.fc2.bias.data
                    svd_decoder.self_attn_layer_norm = layer.self_attn_layer_norm
                    svd_decoder.final_layer_norm = layer.final_layer_norm
                    layers[i] = svd_decoder
            else:
                if "q_proj" in name:
                    svd_attn.q_u_proj.weight.data = svd_u
                    svd_attn.q_v_proj.weight.data = svd_v
                elif "k_proj" in name:
                    svd_attn.k_u_proj.weight.data = svd_u
                    svd_attn.k_v_proj.weight.data = svd_v
                elif "v_proj" in name:
                    svd_attn.v_u_proj.weight.data = svd_u
                    svd_attn.v_v_proj.weight.data = svd_v
                elif "o_proj" in name:
                    svd_attn.o_u_proj.weight.data = svd_u
                    svd_attn.o_v_proj.weight.data = svd_v
                    layer.self_attn =  svd_attn
                elif "gate_proj" in name:
                    svd_mlp.gate_u_proj.weight.data = svd_u
                    svd_mlp.gate_v_proj.weight.data = svd_v
                elif "down_proj" in name:
                    svd_mlp.down_u_proj.weight.data = svd_u
                    svd_mlp.down_v_proj.weight.data = svd_v
                elif "up_proj" in name:
                    svd_mlp.up_u_proj.weight.data = svd_u
                    svd_mlp.up_v_proj.weight.data = svd_v
                    layer.mlp = svd_mlp
            W = W_scale = scaling_matrix_inv = scaling_diag_matrix = U = S = VT  = truc_s = truc_u = truc_v = sqrtSigma = None
            del  W, W_scale, scaling_matrix_inv, scaling_diag_matrix, U, S, VT, truc_s, truc_u, truc_v, sqrtSigma
        del layer
        torch.cuda.empty_cache()


def _compute_whitened_svd(weight, scaling_diag_matrix, dev):
    W = weight.data.float().to(dev)
    dtype = W.dtype
    scaling_diag_matrix = scaling_diag_matrix.to(dev)
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
    return SimpleNamespace(
        W=W,
        dtype=dtype,
        scaling_diag_matrix=scaling_diag_matrix,
        scaling_matrix_inv=scaling_matrix_inv,
        W_scale=W_scale,
        U=U,
        S=S,
        VT=VT,
    )


def _build_factorized_weight(U, S, VT, scaling_matrix_inv, selected_idx, dtype):
    selected_idx = selected_idx.to(U.device)
    selected_sigma = S.index_select(0, selected_idx)
    selected_u = U.index_select(1, selected_idx)
    selected_v = torch.matmul(VT.index_select(0, selected_idx), scaling_matrix_inv)
    sqrt_sigma = torch.diag(torch.sqrt(selected_sigma))
    svd_u = torch.matmul(selected_u, sqrt_sigma).cpu().to(dtype)
    svd_v = torch.matmul(sqrt_sigma, selected_v).cpu().to(dtype)
    return svd_u, svd_v


def _get_target_rank(weight, ratio):
    return int(weight.shape[0] * weight.shape[1] * ratio / (weight.shape[0] + weight.shape[1]))


@torch.no_grad()
def _prepare_task_aware_candidates(model_name, model, profiling_mat, ratio, candidate_extra, dev):
    if 'opt' in model_name:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers
    candidate_map = {}
    print("Preparing whitened SVD candidates for task-aware selection...")
    for i in tqdm(range(len(layers))):
        subset = find_layers(layers[i])
        for name in subset:
            svd_ctx = _compute_whitened_svd(subset[name].weight, profiling_mat[i][name], dev)
            target_rank = _get_target_rank(subset[name].weight, ratio)
            full_rank = svd_ctx.S.shape[0]
            candidate_rank = min(full_rank, target_rank + candidate_extra)
            candidate_map[subset[name]] = {
                "layer_idx": i,
                "name": name,
                "target_rank": target_rank,
                "candidate_rank": candidate_rank,
                "sigma": svd_ctx.S[:candidate_rank].cpu(),
                "u": svd_ctx.U[:, :candidate_rank].cpu(),
                "v": torch.matmul(svd_ctx.VT[:candidate_rank, :], svd_ctx.scaling_matrix_inv).cpu(),
                "sum_phi": torch.zeros(candidate_rank, dtype=torch.float64),
                "sum_phi_sq": torch.zeros(candidate_rank, dtype=torch.float64),
                "sum_phi_outer": torch.zeros(candidate_rank, candidate_rank, dtype=torch.float64),
                "count": 0,
            }
            del svd_ctx
            torch.cuda.empty_cache()
    return candidate_map


def _collect_task_aware_stats(model, calibration_loader, candidate_map, dev):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    model = model.to(dev)
    model.eval()
    model.zero_grad(set_to_none=True)
    handles = []

    def forward_hook(module, inputs, output):
        inp = inputs[0].detach().float()
        info = candidate_map[module]
        v = info["v"].to(inp.device)
        module._task_aware_proj = torch.matmul(inp, v.t())

    def backward_hook(module, grad_input, grad_output):
        proj = module._task_aware_proj
        grad = grad_output[0].detach().float()
        info = candidate_map[module]
        u = info["u"].to(grad.device)
        phi = torch.matmul(grad, u) * proj
        phi = phi.reshape(-1, info["candidate_rank"])
        info["sum_phi"] += phi.sum(dim=0).double().cpu()
        info["sum_phi_sq"] += phi.pow(2).sum(dim=0).double().cpu()
        info["sum_phi_outer"] += torch.matmul(phi.t(), phi).double().cpu()
        info["count"] += phi.shape[0]
        del module._task_aware_proj

    for module in candidate_map:
        handles.append(module.register_forward_hook(forward_hook))
        handles.append(module.register_full_backward_hook(backward_hook))

    print("Collecting task-aware singular saliency statistics...")
    for batch in tqdm(calibration_loader):
        model.zero_grad(set_to_none=True)
        if isinstance(batch, dict):
            inputs = {k: v.to(dev) for k, v in batch.items()}
            labels = inputs["input_ids"]
            outputs = model(**inputs, labels=labels, use_cache=False)
        else:
            input_ids = batch[0].to(dev)
            outputs = model(input_ids=input_ids, labels=input_ids, use_cache=False)
        outputs.loss.backward()
        del outputs
        torch.cuda.empty_cache()

    for handle in handles:
        handle.remove()
    model.zero_grad(set_to_none=True)
    model.config.use_cache = use_cache
    model = model.cpu()
    torch.cuda.empty_cache()


def _select_task_aware_indices(info, selection_method):
    if info["target_rank"] == 0:
        return torch.empty(0, dtype=torch.long)
    if info["candidate_rank"] <= info["target_rank"]:
        return torch.arange(info["candidate_rank"], dtype=torch.long)

    count = max(info["count"], 1)
    sigma = info["sigma"].float()

    if selection_method == "task_aware_diag":
        h = (info["sum_phi"] / count).float()
        fii = (info["sum_phi_sq"] / count).float()
        saliency = -h * sigma + 0.5 * sigma.pow(2) * fii
    elif selection_method == "task_aware_obs":
        fisher = (info["sum_phi_outer"] / count).float()
        eye = torch.eye(fisher.shape[0], dtype=fisher.dtype, device=fisher.device)
        damping = 1e-6 * fisher.diag().mean().clamp_min(1.0)
        fisher_inv = torch.linalg.inv(fisher + damping * eye)
        saliency = 0.5 * sigma.pow(2) / fisher_inv.diag().clamp_min(1e-12)
    else:
        raise ValueError(f"Unknown selection_method: {selection_method}")

    selected_idx = torch.topk(saliency, k=info["target_rank"], largest=True).indices
    selected_idx = selected_idx[torch.argsort(saliency.index_select(0, selected_idx), descending=True)]
    return selected_idx


def whitening_task_aware(model_name, model, profiling_mat, ratio, calibration_loader, candidate_extra, dev, selection_method):
    model.eval()
    candidate_map = _prepare_task_aware_candidates(model_name, model, profiling_mat, ratio, candidate_extra, dev)
    _collect_task_aware_stats(model, calibration_loader, candidate_map, dev)
    if 'opt' in model_name:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers
    print("Applying task-aware singular selection...")
    for i in tqdm(range(len(layers))):
        layer = layers[i]
        subset = find_layers(layer)
        if is_llama_family(model_name):
            svd_attn = SVD_LlamaAttention(config=model.config, ratio=ratio)
            svd_mlp = SVD_LlamaMLP(hidden_size=layer.hidden_size, intermediate_size=model.config.intermediate_size, hidden_act=model.config.hidden_act, ratio=ratio)
        elif is_mistral_family(model_name):
            svd_attn = SVD_MistralAttention(config=model.config, ratio=ratio)
            svd_mlp = SVD_MistralMLP(config=model.config, ratio=ratio)
        elif is_opt_model(model_name):
            svd_decoder = SVDOPTDecoderLayer(model.config, ratio=ratio)
        for name in subset:
            info = candidate_map[subset[name]]
            svd_ctx = _compute_whitened_svd(subset[name].weight, profiling_mat[i][name], dev)
            selected_idx = _select_task_aware_indices(info, selection_method)
            svd_u, svd_v = _build_factorized_weight(
                svd_ctx.U,
                svd_ctx.S,
                svd_ctx.VT,
                svd_ctx.scaling_matrix_inv,
                selected_idx,
                svd_ctx.dtype,
            )
            if is_opt_model(model_name):
                if "q_proj" in name:
                    svd_decoder.self_attn.q_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.q_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.q_u_proj.bias.data = layer.self_attn.q_proj.bias.data
                elif "k_proj" in name:
                    svd_decoder.self_attn.k_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.k_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.k_u_proj.bias.data = layer.self_attn.k_proj.bias.data
                elif "v_proj" in name:
                    svd_decoder.self_attn.v_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.v_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.v_u_proj.bias.data = layer.self_attn.v_proj.bias.data
                elif "out_proj" in name:
                    svd_decoder.self_attn.out_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.out_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.out_u_proj.bias.data = layer.self_attn.out_proj.bias.data
                elif "fc1" in name:
                    svd_decoder.fc1_u_proj.weight.data = svd_u
                    svd_decoder.fc1_v_proj.weight.data = svd_v
                    svd_decoder.fc1_u_proj.bias.data = layer.fc1.bias.data
                elif "fc2" in name:
                    svd_decoder.fc2_u_proj.weight.data = svd_u
                    svd_decoder.fc2_v_proj.weight.data = svd_v
                    svd_decoder.fc2_u_proj.bias.data = layer.fc2.bias.data
                    svd_decoder.self_attn_layer_norm = layer.self_attn_layer_norm
                    svd_decoder.final_layer_norm = layer.final_layer_norm
                    layers[i] = svd_decoder
            else:
                if "q_proj" in name:
                    svd_attn.q_u_proj.weight.data = svd_u
                    svd_attn.q_v_proj.weight.data = svd_v
                elif "k_proj" in name:
                    svd_attn.k_u_proj.weight.data = svd_u
                    svd_attn.k_v_proj.weight.data = svd_v
                elif "v_proj" in name:
                    svd_attn.v_u_proj.weight.data = svd_u
                    svd_attn.v_v_proj.weight.data = svd_v
                elif "o_proj" in name:
                    svd_attn.o_u_proj.weight.data = svd_u
                    svd_attn.o_v_proj.weight.data = svd_v
                    layer.self_attn =  svd_attn
                elif "gate_proj" in name:
                    svd_mlp.gate_u_proj.weight.data = svd_u
                    svd_mlp.gate_v_proj.weight.data = svd_v
                elif "down_proj" in name:
                    svd_mlp.down_u_proj.weight.data = svd_u
                    svd_mlp.down_v_proj.weight.data = svd_v
                elif "up_proj" in name:
                    svd_mlp.up_u_proj.weight.data = svd_u
                    svd_mlp.up_v_proj.weight.data = svd_v
                    layer.mlp = svd_mlp
            del svd_ctx
            torch.cuda.empty_cache()
        del layer
        torch.cuda.empty_cache()


class buffered_subspace_refit:
    def __init__(self, layer, scaling_diag_matrix, ratio, buffer_size, reg_lambda, name):
        self.layer = layer
        self.name = name
        self.dev = self.layer.weight.device
        self.dtype = self.layer.weight.data.dtype

        svd_ctx = _compute_whitened_svd(self.layer.weight, scaling_diag_matrix, self.dev)
        self.target_rank = _get_target_rank(self.layer.weight, ratio)
        self.buffer_rank = min(svd_ctx.S.shape[0], self.target_rank + buffer_size)

        self.u_bar = svd_ctx.U[:, :self.buffer_rank].float()
        self.v_bar_t_s_inv = torch.matmul(svd_ctx.VT[:self.buffer_rank, :], svd_ctx.scaling_matrix_inv).float()
        self.reg_lambda = float(reg_lambda)

        self.c0 = torch.zeros(self.buffer_rank, self.buffer_rank, device=self.dev, dtype=torch.float32)
        if self.target_rank > 0:
            self.c0[:self.target_rank, :self.target_rank] = torch.diag(svd_ctx.S[:self.target_rank].float())

        self.tzt = torch.zeros(self.buffer_rank, self.buffer_rank, device=self.dev, dtype=torch.float32)
        self.zzt = torch.zeros(self.buffer_rank, self.buffer_rank, device=self.dev, dtype=torch.float32)

        del svd_ctx
        torch.cuda.empty_cache()

    def add_batch(self, inp, out):
        inps = inp.view(inp.shape[0] * inp.shape[1], inp.shape[2]).float()
        outs = out.view(out.shape[0] * out.shape[1], out.shape[2]).float()
        z = torch.matmul(inps, self.v_bar_t_s_inv.t())
        t = torch.matmul(outs, self.u_bar)
        self.tzt += torch.matmul(t.t(), z)
        self.zzt += torch.matmul(z.t(), z)

    def solve(self):
        if self.target_rank == 0:
            empty_u = torch.zeros(self.layer.weight.shape[0], 0, device="cpu", dtype=self.dtype)
            empty_v = torch.zeros(0, self.layer.weight.shape[1], device="cpu", dtype=self.dtype)
            return empty_u, empty_v

        if self.buffer_rank == self.target_rank and self.reg_lambda == 0:
            sqrt_sigma = torch.sqrt(torch.diag(self.c0[:self.target_rank, :self.target_rank]))
            svd_u = torch.matmul(self.u_bar[:, :self.target_rank], sqrt_sigma).cpu().to(self.dtype)
            svd_v = torch.matmul(sqrt_sigma, self.v_bar_t_s_inv[:self.target_rank, :]).cpu().to(self.dtype)
            return svd_u, svd_v

        eye = torch.eye(self.buffer_rank, device=self.dev, dtype=torch.float32)
        G = self.zzt + self.reg_lambda * eye
        eigvals, eigvecs = torch.linalg.eigh(G.double())
        eigvals = eigvals.clamp_min(1e-8)
        g_inv_sqrt = torch.matmul(eigvecs, torch.matmul(torch.diag(eigvals.rsqrt()), eigvecs.t())).float()

        B = torch.matmul(self.tzt + self.reg_lambda * self.c0, g_inv_sqrt)
        P, S, VT = torch.linalg.svd(B, full_matrices=False)

        pk = P[:, :self.target_rank]
        sqrt_lambda = torch.diag(torch.sqrt(S[:self.target_rank]))
        qk_t = VT[:self.target_rank, :]

        svd_u = torch.matmul(torch.matmul(self.u_bar, pk), sqrt_lambda).cpu().to(self.dtype)
        svd_v = torch.matmul(torch.matmul(sqrt_lambda, qk_t), torch.matmul(g_inv_sqrt, self.v_bar_t_s_inv)).cpu().to(self.dtype)
        return svd_u, svd_v


class singular_expert_merge:
    def __init__(self, layer, scaling_diag_matrix, ratio, merge_budget, top_candidates, reg_lambda, name):
        self.layer = layer
        self.name = name
        self.dev = self.layer.weight.device
        self.dtype = self.layer.weight.data.dtype
        self.reg_lambda = float(reg_lambda)

        svd_ctx = _compute_whitened_svd(self.layer.weight, scaling_diag_matrix, self.dev)
        self.target_rank = _get_target_rank(self.layer.weight, ratio)
        self.tail_rank = min(max(svd_ctx.S.shape[0] - self.target_rank, 0), merge_budget)
        self.top_candidates = min(self.target_rank, top_candidates)

        if self.target_rank > 0:
            sqrt_sigma_k = torch.diag(torch.sqrt(svd_ctx.S[:self.target_rank].float()))
            self.a_k = torch.matmul(svd_ctx.U[:, :self.target_rank].float(), sqrt_sigma_k)
            self.b_k = torch.matmul(sqrt_sigma_k, torch.matmul(svd_ctx.VT[:self.target_rank, :], svd_ctx.scaling_matrix_inv)).float()
        else:
            self.a_k = torch.zeros(self.layer.weight.shape[0], 0, device=self.dev, dtype=torch.float32)
            self.b_k = torch.zeros(0, self.layer.weight.shape[1], device=self.dev, dtype=torch.float32)

        if self.tail_rank > 0:
            tail_slice = slice(self.target_rank, self.target_rank + self.tail_rank)
            sqrt_sigma_r = torch.diag(torch.sqrt(svd_ctx.S[tail_slice].float()))
            self.a_r = torch.matmul(svd_ctx.U[:, tail_slice].float(), sqrt_sigma_r)
            self.b_r = torch.matmul(sqrt_sigma_r, torch.matmul(svd_ctx.VT[tail_slice, :], svd_ctx.scaling_matrix_inv)).float()
        else:
            self.a_r = torch.zeros(self.layer.weight.shape[0], 0, device=self.dev, dtype=torch.float32)
            self.b_r = torch.zeros(0, self.layer.weight.shape[1], device=self.dev, dtype=torch.float32)

        del svd_ctx
        torch.cuda.empty_cache()

    def fit(self, inp, out):
        if self.target_rank == 0:
            empty_u = torch.zeros(self.layer.weight.shape[0], 0, device="cpu", dtype=self.dtype)
            empty_v = torch.zeros(0, self.layer.weight.shape[1], device="cpu", dtype=self.dtype)
            return empty_u, empty_v

        if self.tail_rank == 0 or self.top_candidates == 0:
            return self.a_k.cpu().to(self.dtype), self.b_k.cpu().to(self.dtype)

        x = inp.view(inp.shape[0] * inp.shape[1], inp.shape[2]).float()
        y = out.view(out.shape[0] * out.shape[1], out.shape[2]).float()

        z_k = torch.matmul(x, self.b_k.t())
        z_r = torch.matmul(x, self.b_r.t())

        z_k_norm = torch.norm(z_k, dim=0).clamp_min(1e-8)
        z_r_norm = torch.norm(z_r, dim=0).clamp_min(1e-8)
        similarity = torch.matmul(z_r.t(), z_k) / torch.outer(z_r_norm, z_k_norm)

        candidate_mask = torch.zeros(self.tail_rank, self.target_rank, device=self.dev, dtype=torch.bool)
        candidate_idx = torch.topk(similarity.abs(), k=self.top_candidates, dim=1).indices
        candidate_mask.scatter_(1, candidate_idx, True)

        P = torch.zeros(self.tail_rank, self.target_rank, device=self.dev, dtype=torch.float32)
        eye_cache = {}
        for j in range(self.tail_rank):
            support = candidate_idx[j]
            z_support = z_k[:, support]
            gram = torch.matmul(z_support.t(), z_support)
            t = gram.shape[0]
            if t not in eye_cache:
                eye_cache[t] = torch.eye(t, device=self.dev, dtype=torch.float32)
            rhs = torch.matmul(z_support.t(), z_r[:, j])
            sol = torch.linalg.solve(gram + self.reg_lambda * eye_cache[t], rhs)
            P[j, support] = sol

        a_hat = self.a_k + torch.matmul(self.a_r, P)
        left_pred = torch.matmul(z_k, a_hat.t())
        residual = y - left_pred

        coeff = torch.linalg.lstsq(a_hat.float(), residual.t().float()).solution
        Q = torch.zeros(self.target_rank, self.tail_rank, device=self.dev, dtype=torch.float32)
        for i in range(self.target_rank):
            support = torch.nonzero(candidate_mask[:, i], as_tuple=False).flatten()
            if support.numel() == 0:
                fallback = torch.topk(similarity[:, i].abs(), k=min(self.tail_rank, self.top_candidates), dim=0).indices
                support = fallback.flatten()
            z_support = z_r[:, support]
            gram = torch.matmul(z_support.t(), z_support)
            t = gram.shape[0]
            if t not in eye_cache:
                eye_cache[t] = torch.eye(t, device=self.dev, dtype=torch.float32)
            rhs = torch.matmul(z_support.t(), coeff[i])
            sol = torch.linalg.solve(gram + self.reg_lambda * eye_cache[t], rhs)
            Q[i, support] = sol

        b_hat = self.b_k + torch.matmul(Q, self.b_r)
        return a_hat.cpu().to(self.dtype), b_hat.cpu().to(self.dtype)


@torch.no_grad()
def whitening_buffered_refit(model_name, model, dataloader, profiling_mat, ratio, dev, buffer_size, reg_lambda):
    print("Start buffered singular subspace refit...")
    use_cache = model.config.use_cache
    model.config.use_cache = False
    if is_opt_model(model_name):
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    else:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
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

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['layer_kwargs'].append(_to_cpu_structure(kwargs))
            cache['i'] += 1
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            prepared_batch = _prepare_model_inputs(batch, model, dev)
            model(prepared_batch[0])
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if is_opt_model(model_name):
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    else:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    layer_kwargs = cache['layer_kwargs']

    for i in tqdm(range(len(layers))):
        layer = layers[i].to(dev)
        subset = find_layers(layer)
        refits = {}
        if is_llama_family(model_name):
            svd_attn = SVD_LlamaAttention(config=model.config, ratio=ratio)
            svd_mlp = SVD_LlamaMLP(hidden_size=layer.hidden_size, intermediate_size=model.config.intermediate_size, hidden_act=model.config.hidden_act, ratio=ratio)
        elif is_mistral_family(model_name):
            svd_attn = SVD_MistralAttention(config=model.config, ratio=ratio)
            svd_mlp = SVD_MistralMLP(config=model.config, ratio=ratio)
        elif is_opt_model(model_name):
            svd_decoder = SVDOPTDecoderLayer(model.config, ratio=ratio)

        for name in subset:
            refits[name] = buffered_subspace_refit(
                subset[name],
                profiling_mat[i][name].to(dev),
                ratio,
                buffer_size,
                reg_lambda,
                name,
            )

        def add_batch(name):
            def tmp(_, inp, out):
                refits[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in refits:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(inps.shape[0]):
            _run_layer_with_kwargs(layer, inps[j].unsqueeze(0), layer_kwargs[j], dev)

        for h in handles:
            h.remove()

        for name in refits:
            svd_u, svd_v = refits[name].solve()
            svd_u, svd_v = svd_u.to(dtype), svd_v.to(dtype)
            if is_opt_model(model_name):
                if "q_proj" in name:
                    svd_decoder.self_attn.q_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.q_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.q_u_proj.bias.data = layer.self_attn.q_proj.bias.data
                elif "k_proj" in name:
                    svd_decoder.self_attn.k_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.k_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.k_u_proj.bias.data = layer.self_attn.k_proj.bias.data
                elif "v_proj" in name:
                    svd_decoder.self_attn.v_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.v_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.v_u_proj.bias.data = layer.self_attn.v_proj.bias.data
                elif "out_proj" in name:
                    svd_decoder.self_attn.out_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.out_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.out_u_proj.bias.data = layer.self_attn.out_proj.bias.data
                elif "fc1" in name:
                    svd_decoder.fc1_u_proj.weight.data = svd_u
                    svd_decoder.fc1_v_proj.weight.data = svd_v
                    svd_decoder.fc1_u_proj.bias.data = layer.fc1.bias.data
                elif "fc2" in name:
                    svd_decoder.fc2_u_proj.weight.data = svd_u
                    svd_decoder.fc2_v_proj.weight.data = svd_v
                    svd_decoder.fc2_u_proj.bias.data = layer.fc2.bias.data
                    svd_decoder.self_attn_layer_norm = layer.self_attn_layer_norm
                    svd_decoder.final_layer_norm = layer.final_layer_norm
                    layer = svd_decoder
            else:
                if "q_proj" in name:
                    svd_attn.q_u_proj.weight.data = svd_u
                    svd_attn.q_v_proj.weight.data = svd_v
                elif "k_proj" in name:
                    svd_attn.k_u_proj.weight.data = svd_u
                    svd_attn.k_v_proj.weight.data = svd_v
                elif "v_proj" in name:
                    svd_attn.v_u_proj.weight.data = svd_u
                    svd_attn.v_v_proj.weight.data = svd_v
                elif "o_proj" in name:
                    svd_attn.o_u_proj.weight.data = svd_u
                    svd_attn.o_v_proj.weight.data = svd_v
                    layer.self_attn = svd_attn
                elif "gate_proj" in name:
                    svd_mlp.gate_u_proj.weight.data = svd_u
                    svd_mlp.gate_v_proj.weight.data = svd_v
                elif "down_proj" in name:
                    svd_mlp.down_u_proj.weight.data = svd_u
                    svd_mlp.down_v_proj.weight.data = svd_v
                elif "up_proj" in name:
                    svd_mlp.up_u_proj.weight.data = svd_u
                    svd_mlp.up_v_proj.weight.data = svd_v
                    layer.mlp = svd_mlp

        layer = layer.to(dev)
        outs = torch.zeros_like(inps)
        for j in range(inps.shape[0]):
            outs[j] = _run_layer_with_kwargs(layer, inps[j].unsqueeze(0), layer_kwargs[j], dev)
        layers[i] = layer.cpu()
        del refits
        torch.cuda.empty_cache()
        inps = outs
        outs = None
        del outs

    model.config.use_cache = use_cache


@torch.no_grad()
def whitening_singular_expert_merge(model_name, model, dataloader, profiling_mat, ratio, dev, merge_budget, top_candidates, reg_lambda):
    print("Start singular expert merging...")
    use_cache = model.config.use_cache
    model.config.use_cache = False
    if is_opt_model(model_name):
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    else:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
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

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['layer_kwargs'].append(_to_cpu_structure(kwargs))
            cache['i'] += 1
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            prepared_batch = _prepare_model_inputs(batch, model, dev)
            model(prepared_batch[0])
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if is_opt_model(model_name):
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    else:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    layer_kwargs = cache['layer_kwargs']

    for i in tqdm(range(len(layers))):
        layer = layers[i].to(dev)
        subset = find_layers(layer)
        mergers = {}
        if is_llama_family(model_name):
            svd_attn = SVD_LlamaAttention(config=model.config, ratio=ratio)
            svd_mlp = SVD_LlamaMLP(hidden_size=layer.hidden_size, intermediate_size=model.config.intermediate_size, hidden_act=model.config.hidden_act, ratio=ratio)
        elif is_mistral_family(model_name):
            svd_attn = SVD_MistralAttention(config=model.config, ratio=ratio)
            svd_mlp = SVD_MistralMLP(config=model.config, ratio=ratio)
        elif is_opt_model(model_name):
            svd_decoder = SVDOPTDecoderLayer(model.config, ratio=ratio)

        for name in subset:
            mergers[name] = singular_expert_merge(
                subset[name],
                profiling_mat[i][name].to(dev),
                ratio,
                merge_budget,
                top_candidates,
                reg_lambda,
                name,
            )

        merged_factors = {}

        def add_batch(name):
            def tmp(_, inp, out):
                merged_factors[name] = mergers[name].fit(inp[0].data, out.data)
            return tmp

        handles = []
        for name in mergers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(inps.shape[0]):
            _run_layer_with_kwargs(layer, inps[j].unsqueeze(0), layer_kwargs[j], dev)

        for h in handles:
            h.remove()

        for name in merged_factors:
            svd_u, svd_v = merged_factors[name]
            svd_u, svd_v = svd_u.to(dtype), svd_v.to(dtype)
            if is_opt_model(model_name):
                if "q_proj" in name:
                    svd_decoder.self_attn.q_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.q_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.q_u_proj.bias.data = layer.self_attn.q_proj.bias.data
                elif "k_proj" in name:
                    svd_decoder.self_attn.k_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.k_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.k_u_proj.bias.data = layer.self_attn.k_proj.bias.data
                elif "v_proj" in name:
                    svd_decoder.self_attn.v_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.v_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.v_u_proj.bias.data = layer.self_attn.v_proj.bias.data
                elif "out_proj" in name:
                    svd_decoder.self_attn.out_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.out_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.out_u_proj.bias.data = layer.self_attn.out_proj.bias.data
                elif "fc1" in name:
                    svd_decoder.fc1_u_proj.weight.data = svd_u
                    svd_decoder.fc1_v_proj.weight.data = svd_v
                    svd_decoder.fc1_u_proj.bias.data = layer.fc1.bias.data
                elif "fc2" in name:
                    svd_decoder.fc2_u_proj.weight.data = svd_u
                    svd_decoder.fc2_v_proj.weight.data = svd_v
                    svd_decoder.fc2_u_proj.bias.data = layer.fc2.bias.data
                    svd_decoder.self_attn_layer_norm = layer.self_attn_layer_norm
                    svd_decoder.final_layer_norm = layer.final_layer_norm
                    layer = svd_decoder
            else:
                if "q_proj" in name:
                    svd_attn.q_u_proj.weight.data = svd_u
                    svd_attn.q_v_proj.weight.data = svd_v
                elif "k_proj" in name:
                    svd_attn.k_u_proj.weight.data = svd_u
                    svd_attn.k_v_proj.weight.data = svd_v
                elif "v_proj" in name:
                    svd_attn.v_u_proj.weight.data = svd_u
                    svd_attn.v_v_proj.weight.data = svd_v
                elif "o_proj" in name:
                    svd_attn.o_u_proj.weight.data = svd_u
                    svd_attn.o_v_proj.weight.data = svd_v
                    layer.self_attn = svd_attn
                elif "gate_proj" in name:
                    svd_mlp.gate_u_proj.weight.data = svd_u
                    svd_mlp.gate_v_proj.weight.data = svd_v
                elif "down_proj" in name:
                    svd_mlp.down_u_proj.weight.data = svd_u
                    svd_mlp.down_v_proj.weight.data = svd_v
                elif "up_proj" in name:
                    svd_mlp.up_u_proj.weight.data = svd_u
                    svd_mlp.up_v_proj.weight.data = svd_v
                    layer.mlp = svd_mlp

        layer = layer.to(dev)
        outs = torch.zeros_like(inps)
        for j in range(inps.shape[0]):
            outs[j] = _run_layer_with_kwargs(layer, inps[j].unsqueeze(0), layer_kwargs[j], dev)
        layers[i] = layer.cpu()
        del mergers, merged_factors
        torch.cuda.empty_cache()
        inps = outs
        outs = None
        del outs

    model.config.use_cache = use_cache


@torch.no_grad()
def whitening_local_update(model_name, model, dataloader, profiling_mat, ratio, dev, direct_update=False):
    print("Start SVD decomposition then update...")
    use_cache = model.config.use_cache
    model.config.use_cache = False
    if is_opt_model(model_name):
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
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['layer_kwargs'].append(_to_cpu_structure(kwargs))
            cache['i'] += 1
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            prepared_batch = _prepare_model_inputs(batch, model, dev)
            model(prepared_batch[0])
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    layer_kwargs = cache['layer_kwargs']
    for i in tqdm(range(len(layers))):
        layer = layers[i].to(dev)
        subset = find_layers(layer)
        gpts = {}
        if is_llama_family(model_name):
            svd_attn = SVD_LlamaAttention(config=model.config, ratio=ratio)
            svd_mlp = SVD_LlamaMLP(hidden_size=layer.hidden_size, intermediate_size=model.config.intermediate_size, hidden_act=model.config.hidden_act, ratio=ratio)
        elif is_mistral_family(model_name):
            svd_attn = SVD_MistralAttention(config=model.config, ratio=ratio)
            svd_mlp = SVD_MistralMLP(config=model.config, ratio=ratio)
        elif is_opt_model(model_name):
            svd_decoder = SVDOPTDecoderLayer(model.config, ratio=ratio)
        for name in subset:
            if profiling_mat is not None:
                scaling_diag_matrix = profiling_mat[i][name].to(dev)
            else: 
                scaling_diag_matrix = None
            gpts[name] = local_update(subset[name], scaling_diag_matrix = scaling_diag_matrix, ratio=ratio, name=name, direct_update=direct_update)
        
        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch_update_u(inp[0].data, out.data)
            return tmp
        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        outs = torch.zeros_like(inps)
        for j in range(inps.shape[0]):
            outs[j] = _run_layer_with_kwargs(layer, inps[j].unsqueeze(0), layer_kwargs[j], dev)
        for h in handles:
            h.remove()
        for name in gpts:
            svd_u, svd_v = gpts[name].fasterprune()
            svd_u, svd_v = svd_u.to(dtype), svd_v.to(dtype)
            if is_opt_model(model_name):
                if "q_proj" in name:
                    svd_decoder.self_attn.q_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.q_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.q_u_proj.bias.data = layer.self_attn.q_proj.bias.data  # the linear layer in OPT has bias, which is different from LLaMA and Mistral
                elif "k_proj" in name:
                    svd_decoder.self_attn.k_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.k_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.k_u_proj.bias.data = layer.self_attn.k_proj.bias.data
                elif "v_proj" in name:
                    svd_decoder.self_attn.v_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.v_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.v_u_proj.bias.data = layer.self_attn.v_proj.bias.data
                elif "out_proj" in name:
                    svd_decoder.self_attn.out_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.out_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.out_u_proj.bias.data = layer.self_attn.out_proj.bias.data
                elif "fc1" in name:
                    svd_decoder.fc1_u_proj.weight.data = svd_u
                    svd_decoder.fc1_v_proj.weight.data = svd_v
                    svd_decoder.fc1_u_proj.bias.data = layer.fc1.bias.data
                elif "fc2" in name:
                    svd_decoder.fc2_u_proj.weight.data = svd_u
                    svd_decoder.fc2_v_proj.weight.data = svd_v
                    svd_decoder.fc2_u_proj.bias.data = layer.fc2.bias.data
                    svd_decoder.self_attn_layer_norm = layer.self_attn_layer_norm
                    svd_decoder.final_layer_norm = layer.final_layer_norm
                    layers[i] = svd_decoder
            else:
                if "q_proj" in name:
                    svd_attn.q_u_proj.weight.data = svd_u
                    svd_attn.q_v_proj.weight.data = svd_v
                elif "k_proj" in name:
                    svd_attn.k_u_proj.weight.data = svd_u
                    svd_attn.k_v_proj.weight.data = svd_v
                elif "v_proj" in name:
                    svd_attn.v_u_proj.weight.data = svd_u
                    svd_attn.v_v_proj.weight.data = svd_v
                elif "o_proj" in name:
                    svd_attn.o_u_proj.weight.data = svd_u
                    svd_attn.o_v_proj.weight.data = svd_v
                    layer.self_attn =  svd_attn
                elif "gate_proj" in name:
                    svd_mlp.gate_u_proj.weight.data = svd_u
                    svd_mlp.gate_v_proj.weight.data = svd_v
                elif "down_proj" in name:
                    svd_mlp.down_u_proj.weight.data = svd_u
                    svd_mlp.down_v_proj.weight.data = svd_v
                elif "up_proj" in name:
                    svd_mlp.up_u_proj.weight.data = svd_u
                    svd_mlp.up_v_proj.weight.data = svd_v
                    layer.mlp = svd_mlp
        layer = layer.to(dev)
        outs = torch.zeros_like(inps)
        for j in range(inps.shape[0]):
            outs[j] = _run_layer_with_kwargs(layer, inps[j].unsqueeze(0), layer_kwargs[j], dev)
        layers[i] = layer.cpu()
        del gpts
        torch.cuda.empty_cache()
        inps = outs
        outs = None
        del outs
    model.config.use_cache = use_cache


class local_update:
    def __init__(self, layer, scaling_diag_matrix, ratio, name, direct_update=False):
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
        num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
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
    parser.add_argument('--dataset', type=str, default='wikitext2',help='Where to extract calibration data from [wikitext2, ptb, c4]')
    parser.add_argument('--whitening_nsamples', type=int, default=256, help='Number of calibration data samples for whitening.')
    parser.add_argument('--updating_nsamples', type=int, default=16, help='Number of calibration data samples for udpating.')
    parser.add_argument('--save_path', type=str, default=None, help='the path to save the compressed model checkpoints.`')
    parser.add_argument('--profiling_mat_path', type=str, default=None, help='Local path to load the profiling matrices`')
    parser.add_argument('--seed',type=int, default=0, help='Seed for sampling the calibration data')
    parser.add_argument('--DEV', type=str, default="cuda", help='device')
    parser.add_argument('--model_seq_len', type=int, default=2048, help='the default sequence length of the LLM')
    parser.add_argument('--eval_batch_size', type=int, default=4, help='inference bactch size')
    parser.add_argument('--gen_seq_len', type=int, default=1024, help='generated sequence len for efficiency evaluation')
    parser.add_argument('--step', type=int, default=4, help='the step to run the compression')
    parser.add_argument('--lora', type=str, default=None, help='the lora updated weight path to run the accuracy evaluation')
    parser.add_argument('--lm_eval_tasks', type=str, default='mmlu,gsm8k,humaneval', help='Comma-separated lm-evaluation-harness tasks.')
    parser.add_argument('--lm_eval_num_fewshot', type=int, default=0, help='Number of few-shot examples for lm-evaluation-harness.')
    parser.add_argument('--lm_eval_limit', type=float, default=None, help='Optional example limit for lm-evaluation-harness.')
    parser.add_argument('--lm_eval_output_path', type=str, default=None, help='Optional JSON output path for lm-evaluation-harness results.')
    parser.add_argument('--selection_method', type=str, default='topk', choices=['topk', 'task_aware_diag', 'task_aware_obs', 'bssr', 'sem'], help='Singular-direction selection rule for whitening compression.')
    parser.add_argument('--selection_nsamples', type=int, default=16, help='Number of calibration samples used to estimate task-aware singular saliency.')
    parser.add_argument('--selection_candidate_extra', type=int, default=32, help='Additional candidate singular directions considered beyond the target rank for task-aware selection.')
    parser.add_argument('--bssr_buffer_size', type=int, default=16, help='Buffered extra singular directions for BSSR.')
    parser.add_argument('--bssr_lambda', type=float, default=1e-2, help='Ridge regularization strength for BSSR.')
    parser.add_argument('--sem_merge_budget', type=int, default=16, help='Boundary tail size used for singular expert merging.')
    parser.add_argument('--sem_top_candidates', type=int, default=4, help='Top retained experts considered as merge candidates per removed expert.')
    parser.add_argument('--sem_lambda', type=float, default=1e-2, help='Ridge regularization strength for singular expert merging.')
    
    args = parser.parse_args()
    args.DEV = resolve_runtime_device(args.DEV)
    args.ratio = 1- args.ratio
    if args.step == 1:
        model, tokenizer = get_model_from_huggingface(model_id=args.model)
        ensure_unsharded_model(model, args.model, args.DEV)
        model = model.eval()
        if args.selection_method in ["task_aware_diag", "task_aware_obs", "bssr", "sem"]:
            model = model.float()
        if args.profiling_mat_path is None:
            cali_white_data = get_calib_train_data(args.dataset, tokenizer, args.whitening_nsamples, seqlen=args.model_seq_len)
            profiling_mat = profle_svdllm_low_resource(args.model, model, cali_white_data, args.DEV)
            # if args.save_path is not None:
            #     torch.save(profiling_mat, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") + '_profiling_'+ args.dataset + '_' + str(args.whitening_nsamples)  + '_' + str(args.seed)+ '.pt')
        else:
            profiling_mat = torch.load(args.profiling_mat_path)
        if args.selection_method in ["task_aware_diag", "task_aware_obs"]:
            selection_loader, _ = get_loaders(args.dataset, nsamples=args.selection_nsamples, seed=args.seed, tokenizer=tokenizer, seqlen=args.model_seq_len)
            whitening_task_aware(
                args.model,
                model,
                profiling_mat,
                args.ratio,
                selection_loader,
                args.selection_candidate_extra,
                args.DEV,
                args.selection_method,
            )
        elif args.selection_method == "bssr":
            dataloader, _ = get_loaders(args.dataset, nsamples=args.selection_nsamples, seed=args.seed, tokenizer=tokenizer, seqlen=args.model_seq_len)
            whitening_buffered_refit(
                args.model,
                model,
                dataloader,
                profiling_mat,
                args.ratio,
                args.DEV,
                args.bssr_buffer_size,
                args.bssr_lambda,
            )
        elif args.selection_method == "sem":
            dataloader, _ = get_loaders(args.dataset, nsamples=args.selection_nsamples, seed=args.seed, tokenizer=tokenizer, seqlen=args.model_seq_len)
            whitening_singular_expert_merge(
                args.model,
                model,
                dataloader,
                profiling_mat,
                args.ratio,
                args.DEV,
                args.sem_merge_budget,
                args.sem_top_candidates,
                args.sem_lambda,
            )
        else:
            whitening(args.model, model, profiling_mat, args.ratio, args.DEV)
        if args.save_path is not None:
            unwrap_model_for_save(model)
            suffix = '_whitening_only_' + args.selection_method + '_' + str(args.ratio) if args.selection_method != 'topk' else '_whitening_only_' + str(args.ratio)
            save_compressed_checkpoint(
                args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") + suffix + '.pt',
                model,
                tokenizer,
                args.model,
            )
    elif args.step == 2:
        model, tokenizer = get_model_from_huggingface(model_id=args.model)
        ensure_unsharded_model(model, args.model, args.DEV)
        dataloader, _ = get_loaders(args.dataset, nsamples=args.updating_nsamples, seed=args.seed, tokenizer=tokenizer, seqlen=args.model_seq_len)
        model = model.eval()
        model = model.float()  # need to set to float
        if args.profiling_mat_path is None:
            cali_white_data = get_calib_train_data(args.dataset, tokenizer, args.whitening_nsamples, seqlen=args.model_seq_len)
            profiling_mat = profle_svdllm_low_resource(args.model, model, cali_white_data, args.DEV)
            if args.save_path is not None:
                torch.save(profiling_mat, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") + '_profiling_'+ args.dataset + '_' + str(args.whitening_nsamples)  + '_' + str(args.seed)+ '.pt')
        else:
            profiling_mat = torch.load(args.profiling_mat_path)
        whitening_local_update(args.model, model, dataloader, profiling_mat, args.ratio, args.DEV)
        if args.save_path is not None:
            unwrap_model_for_save(model)
            save_compressed_checkpoint(
                args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") +'_whitening_then_update_' + str(args.ratio) + '.pt',
                model,
                tokenizer,
                args.model,
            )
    elif args.step == 3:
        model, tokenizer = get_model_from_huggingface(args.model)
        ensure_unsharded_model(model, args.model, args.DEV)
        model = model.eval()
        model = model.float()
        dataloader, _ = get_loaders(args.dataset, nsamples=args.updating_nsamples, seed=args.seed, tokenizer=tokenizer, seqlen=args.model_seq_len)
        whitening_local_update(model_name=args.model, model=model, dataloader=dataloader, profiling_mat=None, ratio=args.ratio, dev=args.DEV, direct_update=True)
        if args.save_path is not None:
            unwrap_model_for_save(model)
            save_compressed_checkpoint(
                args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") +'_update_only_' + str(args.ratio) + '.pt',
                model,
                tokenizer,
                args.model,
            )
    elif args.step >= 4:
        print(f"evaluating {args.model_path}...")
        if args.model_path == "original":
            model, tokenizer = get_model_from_huggingface(
                args.model,
                device_map="cpu",
                torch_dtype=torch.float16,
            )
        else:
            model, tokenizer = get_model_from_local(
                args.model_path,
                base_model_id=args.model,
                reconstruct=True,
                device_map="cpu",
                torch_dtype=torch.float16,
            )
            if args.lora is not None:
                from utils.peft import PeftModel
                model = PeftModel.from_pretrained(
                    model,
                    args.lora,
                    torch_dtype=torch.float16,
                )
                model = model.merge_and_unload()
                unwrap_model_for_save(model)
                save_compressed_checkpoint(args.lora + '/merge.pt', model, tokenizer, args.model)
        model.eval()
        model = model.float()
        model = model.to(args.DEV)
        if args.step == 4:
            ppl_eval(model, tokenizer, datasets=['wikitext2'], model_seq_len=args.model_seq_len, batch_size=args.eval_batch_size, device=args.DEV)
        elif args.step == 5:
            eff_eval(model, tokenizer, generated_len=args.gen_seq_len, batch_size=args.eval_batch_size, device=args.DEV)
        elif args.step == 6:
            lm_harness_eval(
                model,
                tokenizer,
                tasks=args.lm_eval_tasks,
                num_fewshot=args.lm_eval_num_fewshot,
                batch_size=args.eval_batch_size,
                device=args.DEV,
                limit=args.lm_eval_limit,
                output_path=args.lm_eval_output_path,
            )
