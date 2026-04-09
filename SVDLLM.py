#coding:utf8
import os
import sys
import argparse
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



@torch.no_grad()
def profle_svdllm(name, model, calib_loader, dev):
    if "llama" in name or "mistral" in name or "vicuna" in name:
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
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.raw_scaling_diag_matrix = 0
            module.register_forward_hook(hook)
    for batch in tqdm(calib_loader):
        batch = {k: v.to(dev) for k, v in batch.items()}
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
        

def profile_bi_svdllm(model_name, model, calib_loader, dev, eps_a=1e-6, eps_b=1e-6, stat_device=None, stat_dtype=torch.float32):
    if "llama" in model_name or "mistral" in model_name or "vicuna" in model_name:
        layers = model.model.layers
    elif "opt" in model_name:
        layers = model.model.decoder.layers
    else:
        raise NotImplementedError(f"Unsupported model type for {model_name}")

    # model = model.to(dev)
    model.eval()
    use_cache = model.config.use_cache
    model.config.use_cache = False
    if stat_device is None:
        stat_device = dev
    print("Start obtaining the bi-whitening matrices...")

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    original_requires_grad = {}
    for name, param in model.named_parameters():
        original_requires_grad[name] = param.requires_grad
        param.requires_grad_(False)

    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)
    input_embedding_hook = model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    def _flatten_feature(tensor):
        tensor = tensor.detach().to(dtype=stat_dtype)
        if tensor.dim() == 2:
            return tensor
        return tensor.reshape(-1, tensor.shape[-1])

    def forward_hook(module, input, output):
        inp = _flatten_feature(input[0]).to(stat_device)
        if isinstance(module.raw_input_cov, int):
            module.raw_input_cov = torch.zeros(
                (inp.shape[1], inp.shape[1]),
                dtype=stat_dtype,
                device=stat_device,
            )
        module.raw_input_cov += inp.transpose(0, 1).matmul(inp)
        module.nsamples += inp.shape[0]

    def backward_hook(module, grad_input, grad_output):
        gout = _flatten_feature(grad_output[0]).to(stat_device)
        if isinstance(module.raw_output_cov, int):
            module.raw_output_cov = torch.zeros(
                (gout.shape[1], gout.shape[1]),
                dtype=stat_dtype,
                device=stat_device,
            )
        module.raw_output_cov += gout.transpose(0, 1).matmul(gout)

    handles = []
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.raw_input_cov = 0
            module.raw_output_cov = 0
            module.nsamples = 0
            handles.append(module.register_forward_hook(forward_hook))
            handles.append(module.register_full_backward_hook(backward_hook))

    for batch in tqdm(calib_loader):
        batch = {k: v.to(dev) for k, v in batch.items()}
        model.zero_grad(set_to_none=True)
        loss = model(**batch, labels=batch["input_ids"]).loss
        loss.backward()

    for handle in handles:
        handle.remove()
    model.zero_grad(set_to_none=True)
    if input_embedding_hook is not None:
        input_embedding_hook.remove()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.config.use_cache = use_cache
    for name, param in model.named_parameters():
        param.requires_grad_(original_requires_grad[name])
    torch.cuda.empty_cache()
    model = model.cpu()

    for i in range(len(layers)):
        subset = find_layers(layers[i])
        for name in subset:
            if not isinstance(subset[name].raw_input_cov, int):
                subset[name].raw_input_cov = subset[name].raw_input_cov.cpu()
            if not isinstance(subset[name].raw_output_cov, int):
                subset[name].raw_output_cov = subset[name].raw_output_cov.cpu()

    profiling_mat = {}
    print("Start bi-whitening factorization...")
    for i in tqdm(range(len(layers))):
        layer_profile = {}
        subset = find_layers(layers[i])
        for name in subset:
            module = subset[name]
            sample_count = max(module.nsamples, 1)
            if isinstance(module.raw_input_cov, int):
                input_cov = eps_a * torch.eye(module.in_features, dtype=stat_dtype)
            else:
                input_cov = module.raw_input_cov / sample_count
            if isinstance(module.raw_output_cov, int):
                output_cov = eps_b * torch.eye(module.out_features, dtype=stat_dtype)
            else:
                output_cov = module.raw_output_cov / sample_count
            input_cov += eps_a * torch.eye(input_cov.shape[0], dtype=input_cov.dtype)
            output_cov += eps_b * torch.eye(output_cov.shape[0], dtype=output_cov.dtype)

            try:
                input_factor = torch.linalg.cholesky(input_cov)
            except Exception:
                print(f"Warning: input covariance is not positive definite at layer {i}:{name}, adding damping.")
                eigenvalues = torch.linalg.eigvalsh(input_cov)
                input_cov += (-eigenvalues[0] + eps_a) * torch.eye(input_cov.shape[0], dtype=input_cov.dtype)
                input_factor = torch.linalg.cholesky(input_cov)

            try:
                output_factor = torch.linalg.cholesky(output_cov)
            except Exception:
                print(f"Warning: output covariance is not positive definite at layer {i}:{name}, adding damping.")
                eigenvalues = torch.linalg.eigvalsh(output_cov)
                output_cov += (-eigenvalues[0] + eps_b) * torch.eye(output_cov.shape[0], dtype=output_cov.dtype)
                output_factor = torch.linalg.cholesky(output_cov)

            layer_profile[name] = {
                "input_factor": input_factor.cpu(),
                "output_factor": output_factor.cpu(),
            }
            module.raw_input_cov = None
            module.raw_output_cov = None
            del module.raw_input_cov, module.raw_output_cov
            torch.cuda.empty_cache()
        profiling_mat[i] = layer_profile
    return profiling_mat


@torch.no_grad()
def profle_svdllm_low_resource(model_name, model, calib_loader, dev):
    if "opt" in model_name:
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
        (len(calib_loader), model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp.cpu()
            cache['i'] += 1
            if cache['attention_mask'] is None:
                cache['attention_mask'] = kwargs['attention_mask'].cpu()
                if "opt" not in model_name:
                    cache['position_ids'] = kwargs['position_ids'].cpu()
            else:
                cache['attention_mask'] = torch.cat((cache['attention_mask'], kwargs['attention_mask'].cpu()), dim=0)
                if "opt" not in model_name:
                    cache['position_ids'] = torch.cat((cache['position_ids'], kwargs['position_ids'].cpu()), dim=0)
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in calib_loader:
        try:
            batch = {k: v.to(dev) for k, v in batch.items()}
            model(**batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "opt" in model_name:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    else:  
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_masks = cache['attention_mask']
    if "opt" not in model_name:
        position_ids = cache['position_ids']
    profiling_mat = {}
    for i in tqdm(range(len(layers))):
        layer_profile = {}
        layer = layers[i].to(dev)
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
            if "opt" not in model_name:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_masks[j].unsqueeze(0).to(dev), position_ids=position_ids[j].unsqueeze(0).to(dev))[0]
            else:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_masks[j].unsqueeze(0).to(dev))[0]
        for h in handles:
            h.remove()
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
        layers[i] = layer.cpu()
        profiling_mat[i] = layer_profile
        inps = outs
        torch.cuda.empty_cache()
    return profiling_mat
     
 
@torch.no_grad()
def bi_whitening(model_name, model, profiling_mat, ratio, dev):
    model.eval()
    if 'opt' in model_name:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers
    print("Start SVD decomposition after bi-whitening...")
    for i in tqdm(range(len(layers))):
        layer = layers[i]
        subset = find_layers(layer)
        if "llama" in model_name or "vicuna" in model_name:
            svd_attn = SVD_LlamaAttention(config=model.config, ratio=ratio).to(dev)
            svd_mlp = SVD_LlamaMLP(hidden_size=layer.hidden_size, intermediate_size=model.config.intermediate_size, hidden_act=model.config.hidden_act, ratio=ratio).to(dev)
        elif "mistral" in model_name:
            svd_attn = SVD_MistralAttention(config=model.config, ratio=ratio).to(dev)
            svd_mlp = SVD_MistralMLP(config=model.config, ratio=ratio).to(dev)
        elif 'opt' in model_name:
            svd_decoder = SVDOPTDecoderLayer(model.config, ratio=ratio).to(dev)
        for name in subset:
            W = subset[name].weight.data.float().to(dev)
            dtype = W.dtype
            input_factor = profiling_mat[i][name]["input_factor"].to(dev).float()
            output_factor = profiling_mat[i][name]["output_factor"].to(dev).float()
            W_scale = torch.matmul(output_factor.transpose(0, 1), torch.matmul(W, input_factor))
            U, S, VT = torch.linalg.svd(W_scale, full_matrices=False)
            num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
            truc_s = S[:num_s_after_trunc]
            truc_u = U[:, :num_s_after_trunc]
            truc_v = VT[:num_s_after_trunc, :]
            sqrtSigma = torch.diag(torch.sqrt(truc_s))
            left_factor = torch.linalg.solve_triangular(
                output_factor.transpose(0, 1),
                torch.matmul(truc_u, sqrtSigma),
                upper=True,
                left=True,
            )
            right_factor = torch.linalg.solve_triangular(
                input_factor.transpose(0, 1),
                torch.matmul(sqrtSigma, truc_v).transpose(0, 1),
                upper=True,
                left=True,
            ).transpose(0, 1)
            svd_u = left_factor.cpu().to(dtype)
            svd_v = right_factor.cpu().to(dtype)
            if 'opt' in model_name:
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
            W = W_scale = input_factor = output_factor = U = S = VT = truc_s = truc_u = truc_v = sqrtSigma = left_factor = right_factor = None
            del W, W_scale, input_factor, output_factor, U, S, VT, truc_s, truc_u, truc_v, sqrtSigma, left_factor, right_factor
        del layer
        torch.cuda.empty_cache()


def sequential_bi_whitening(model_name, model, calib_loader, ratio, dev, eps_a=1e-6, eps_b=1e-6, stat_device=None, stat_dtype=torch.float32):
    if stat_device is None:
        stat_device = dev
    model.eval()
    use_cache = model.config.use_cache
    model.config.use_cache = False

    if 'opt' in model_name:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if model.model.decoder.final_layer_norm is not None:
            model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out is not None:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    else:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    original_requires_grad = {}
    for name, param in model.named_parameters():
        original_requires_grad[name] = param.requires_grad
        param.requires_grad_(False)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (len(calib_loader), model.seqlen, model.config.hidden_size), dtype=dtype, device="cpu"
    )
    labels = torch.zeros((len(calib_loader), model.seqlen), dtype=torch.long, device="cpu")
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp.detach().cpu()
            labels[cache['i']] = cache['current_labels']
            cache['i'] += 1
            if cache['attention_mask'] is None:
                cache['attention_mask'] = kwargs['attention_mask'].detach().cpu()
                if "opt" not in model_name:
                    cache['position_ids'] = kwargs['position_ids'].detach().cpu()
            else:
                cache['attention_mask'] = torch.cat((cache['attention_mask'], kwargs['attention_mask'].detach().cpu()), dim=0)
                if "opt" not in model_name:
                    cache['position_ids'] = torch.cat((cache['position_ids'], kwargs['position_ids'].detach().cpu()), dim=0)
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in calib_loader:
        try:
            batch = {k: v.to(dev) for k, v in batch.items()}
            cache['current_labels'] = batch["input_ids"].detach().cpu()
            model(**batch, labels=batch["input_ids"])
        except ValueError:
            pass
    layers[0] = layers[0].module

    if 'opt' in model_name:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    else:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    attention_masks = cache['attention_mask']
    position_ids = cache['position_ids'] if "opt" not in model_name else None
    profiling_mat = {}
    loss_fct = torch.nn.CrossEntropyLoss()

    def _flatten_feature(tensor):
        tensor = tensor.detach().to(dtype=stat_dtype)
        if tensor.dim() == 2:
            return tensor
        return tensor.reshape(-1, tensor.shape[-1])

    print("Start sequential bi-whitened SVD...")
    for i in tqdm(range(len(layers))):
        layer = layers[i].to(dev)
        subset = find_layers(layer)

        def forward_hook(module, input, output):
            inp = _flatten_feature(input[0]).to(stat_device)
            if isinstance(module.raw_input_cov, int):
                module.raw_input_cov = torch.zeros((inp.shape[1], inp.shape[1]), dtype=stat_dtype, device=stat_device)
            module.raw_input_cov += inp.transpose(0, 1).matmul(inp)
            module.nsamples += inp.shape[0]

        def backward_hook(module, grad_input, grad_output):
            gout = _flatten_feature(grad_output[0]).to(stat_device)
            if isinstance(module.raw_output_cov, int):
                module.raw_output_cov = torch.zeros((gout.shape[1], gout.shape[1]), dtype=stat_dtype, device=stat_device)
            module.raw_output_cov += gout.transpose(0, 1).matmul(gout)

        handles = []
        for name in subset:
            subset[name].raw_input_cov = 0
            subset[name].raw_output_cov = 0
            subset[name].nsamples = 0
            handles.append(subset[name].register_forward_hook(forward_hook))
            handles.append(subset[name].register_full_backward_hook(backward_hook))

        for j in range(inps.shape[0]):
            model.zero_grad(set_to_none=True)
            hidden_states = inps[j].unsqueeze(0).to(dev).detach().requires_grad_(True)
            attention_mask = attention_masks[j].unsqueeze(0).to(dev)
            if "opt" not in model_name:
                position_id = position_ids[j].unsqueeze(0).to(dev)
                hidden_states = layer(hidden_states, attention_mask=attention_mask, position_ids=position_id)[0]
            else:
                hidden_states = layer(hidden_states, attention_mask=attention_mask)[0]

            for k in range(i + 1, len(layers)):
                suffix_layer = layers[k].to(dev)
                if "opt" not in model_name:
                    hidden_states = suffix_layer(hidden_states, attention_mask=attention_mask, position_ids=position_id)[0]
                else:
                    hidden_states = suffix_layer(hidden_states, attention_mask=attention_mask)[0]
                layers[k] = suffix_layer.cpu()
                torch.cuda.empty_cache()

            if 'opt' in model_name:
                if model.model.decoder.final_layer_norm is not None:
                    hidden_states = model.model.decoder.final_layer_norm(hidden_states)
                if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out is not None:
                    hidden_states = model.model.decoder.project_out(hidden_states)
            else:
                hidden_states = model.model.norm(hidden_states)

            logits = model.lm_head(hidden_states)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[j].unsqueeze(0).to(dev)[:, 1:].contiguous()
            loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
            loss.backward()
            model.zero_grad(set_to_none=True)

            hidden_states = logits = shift_logits = shift_labels = loss = None
            del hidden_states, logits, shift_logits, shift_labels, loss
            torch.cuda.empty_cache()

        for h in handles:
            h.remove()

        layer_profile = {}
        for name in subset:
            sample_count = max(subset[name].nsamples, 1)
            input_cov = subset[name].raw_input_cov / sample_count
            output_cov = subset[name].raw_output_cov / sample_count
            input_cov = input_cov.cpu()
            output_cov = output_cov.cpu()
            input_cov += eps_a * torch.eye(input_cov.shape[0], dtype=input_cov.dtype)
            output_cov += eps_b * torch.eye(output_cov.shape[0], dtype=output_cov.dtype)
            try:
                input_factor = torch.linalg.cholesky(input_cov)
            except Exception:
                eigenvalues = torch.linalg.eigvalsh(input_cov)
                input_cov += (-eigenvalues[0] + eps_a) * torch.eye(input_cov.shape[0], dtype=input_cov.dtype)
                input_factor = torch.linalg.cholesky(input_cov)
            try:
                output_factor = torch.linalg.cholesky(output_cov)
            except Exception:
                eigenvalues = torch.linalg.eigvalsh(output_cov)
                output_cov += (-eigenvalues[0] + eps_b) * torch.eye(output_cov.shape[0], dtype=output_cov.dtype)
                output_factor = torch.linalg.cholesky(output_cov)
            layer_profile[name] = {
                "input_factor": input_factor,
                "output_factor": output_factor,
            }
            subset[name].raw_input_cov = None
            subset[name].raw_output_cov = None
            del subset[name].raw_input_cov, subset[name].raw_output_cov
        profiling_mat[i] = layer_profile

        if "llama" in model_name or "vicuna" in model_name:
            svd_attn = SVD_LlamaAttention(config=model.config, ratio=ratio).to(dev)
            svd_mlp = SVD_LlamaMLP(hidden_size=layer.hidden_size, intermediate_size=model.config.intermediate_size, hidden_act=model.config.hidden_act, ratio=ratio).to(dev)
        elif "mistral" in model_name:
            svd_attn = SVD_MistralAttention(config=model.config, ratio=ratio).to(dev)
            svd_mlp = SVD_MistralMLP(config=model.config, ratio=ratio).to(dev)
        elif 'opt' in model_name:
            svd_decoder = SVDOPTDecoderLayer(model.config, ratio=ratio).to(dev)

        for name in subset:
            W = subset[name].weight.data.float().to(dev)
            weight_dtype = W.dtype
            input_factor = layer_profile[name]["input_factor"].to(dev).float()
            output_factor = layer_profile[name]["output_factor"].to(dev).float()
            W_scale = torch.matmul(output_factor.transpose(0, 1), torch.matmul(W, input_factor))
            U, S, VT = torch.linalg.svd(W_scale, full_matrices=False)
            num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
            truc_s = S[:num_s_after_trunc]
            truc_u = U[:, :num_s_after_trunc]
            truc_v = VT[:num_s_after_trunc, :]
            sqrtSigma = torch.diag(torch.sqrt(truc_s))
            left_factor = torch.linalg.solve_triangular(
                output_factor.transpose(0, 1),
                torch.matmul(truc_u, sqrtSigma),
                upper=True,
                left=True,
            )
            right_factor = torch.linalg.solve_triangular(
                input_factor.transpose(0, 1),
                torch.matmul(sqrtSigma, truc_v).transpose(0, 1),
                upper=True,
                left=True,
            ).transpose(0, 1)
            svd_u = left_factor.to(weight_dtype)
            svd_v = right_factor.to(weight_dtype)

            if 'opt' in model_name:
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
                    layers[i] = layer
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

        outs = torch.zeros_like(inps)
        for j in range(inps.shape[0]):
            hidden_states = inps[j].unsqueeze(0).to(dev)
            attention_mask = attention_masks[j].unsqueeze(0).to(dev)
            if "opt" not in model_name:
                position_id = position_ids[j].unsqueeze(0).to(dev)
                outs[j] = layer(hidden_states, attention_mask=attention_mask, position_ids=position_id)[0].detach().cpu()
            else:
                outs[j] = layer(hidden_states, attention_mask=attention_mask)[0].detach().cpu()

        layers[i] = layer.cpu()
        inps = outs
        torch.cuda.empty_cache()

    if 'opt' in model_name:
        if model.model.decoder.final_layer_norm is not None:
            model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.cpu()
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out is not None:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    else:
        model.model.norm = model.model.norm.cpu()
    model.lm_head = model.lm_head.cpu()
    model.config.use_cache = use_cache
    for name, param in model.named_parameters():
        param.requires_grad_(original_requires_grad[name])
    return profiling_mat


@torch.no_grad()
def whitening(model_name, model, profiling_mat, ratio, dev):
    model.eval()
    if 'opt' in model_name:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers
    print("Start SVD decomposition after whitening...")
    for i in tqdm(range(len(layers))):
        layer = layers[i]
        subset = find_layers(layer)
        #### Replace Attn, MLP ####
        if "llama" in model_name or "vicuna" in model_name:
            svd_attn = SVD_LlamaAttention(config=model.config, ratio=ratio)
            svd_mlp = SVD_LlamaMLP(hidden_size=layer.hidden_size, intermediate_size=model.config.intermediate_size, hidden_act=model.config.hidden_act, ratio=ratio)
        elif "mistral" in model_name:
            svd_attn = SVD_MistralAttention(config=model.config, ratio=ratio)
            svd_mlp = SVD_MistralMLP(config=model.config, ratio=ratio)
        elif 'opt' in model_name:
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
            if 'opt' in model_name:
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


@torch.no_grad()
def whitening_local_update(model_name, model, dataloader, profiling_mat, ratio, dev, direct_update=False):
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
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            if cache['attention_mask'] is None:
                cache['attention_mask'] = kwargs['attention_mask']
                if "opt" not in model_name:
                    cache['position_ids'] = kwargs['position_ids']
            else:
                cache['attention_mask'] = torch.cat((cache['attention_mask'], kwargs['attention_mask']), dim=0)
                if "opt" not in model_name:
                    cache['position_ids'] = torch.cat((cache['position_ids'], kwargs['position_ids']), dim=0)
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
    attention_masks = cache['attention_mask']
    if "opt" not in model_name:
        position_ids = cache['position_ids']
    for i in tqdm(range(len(layers))):
        layer = layers[i].to(dev)
        subset = find_layers(layer)
        gpts = {}
        if "llama" in model_name or "vicuna" in model_name:
            svd_attn = SVD_LlamaAttention(config=model.config, ratio=ratio)
            svd_mlp = SVD_LlamaMLP(hidden_size=layer.hidden_size, intermediate_size=model.config.intermediate_size, hidden_act=model.config.hidden_act, ratio=ratio)
        elif "mistral" in model_name:
            svd_attn = SVD_MistralAttention(config=model.config, ratio=ratio)
            svd_mlp = SVD_MistralMLP(config=model.config, ratio=ratio)
        elif 'opt' in model_name:
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
        if "opt" not in model_name:
            outs = layer(inps, attention_mask=attention_masks, position_ids=position_ids)[0]
        else:
            outs = layer(inps, attention_mask=attention_masks)[0]
        for h in handles:
            h.remove()
        for name in gpts:
            svd_u, svd_v = gpts[name].fasterprune()
            svd_u, svd_v = svd_u.to(dtype), svd_v.to(dtype)
            if 'opt' in model_name:
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
        if "opt" not in model_name:
            outs = layer(inps, attention_mask=attention_masks, position_ids=position_ids)[0]
        else:
            outs = layer(inps, attention_mask=attention_masks)[0]
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
    parser.add_argument('--bi_whitening', action='store_true', help='use bi-whitened SVD with both input and output-gradient curvature statistics')
    parser.add_argument('--curvature_eps_a', type=float, default=1e-6, help='input covariance damping for bi-whitened SVD')
    parser.add_argument('--curvature_eps_b', type=float, default=1e-6, help='output-gradient covariance damping for bi-whitened SVD')
    parser.add_argument('--curvature_stat_device', type=str, default='cuda', help='device for accumulating bi-whitening statistics: cuda or cpu')
    parser.add_argument('--curvature_stat_dtype', type=str, default='float32', choices=['float32', 'float64'], help='dtype for accumulating bi-whitening statistics')
    
    args = parser.parse_args()
    args.ratio = 1- args.ratio
    if args.step == 1:
        model, tokenizer = get_model_from_huggingface(model_id=args.model)
        model = model.eval()
        if args.profiling_mat_path is None:
            cali_white_data = get_calib_train_data(args.dataset, tokenizer, args.whitening_nsamples, seqlen=args.model_seq_len)
            if args.bi_whitening:
                profiling_mat = sequential_bi_whitening(
                    args.model,
                    model,
                    cali_white_data,
                    args.ratio,
                    args.DEV,
                    eps_a=args.curvature_eps_a,
                    eps_b=args.curvature_eps_b,
                    stat_device=args.curvature_stat_device,
                    stat_dtype=torch.float32 if args.curvature_stat_dtype == 'float32' else torch.float64,
                )
            else:
                profiling_mat = profle_svdllm_low_resource(args.model, model, cali_white_data, args.DEV)
            if args.save_path is not None:
                torch.save(profiling_mat, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") + '_profiling_'+ args.dataset + '_' + str(args.whitening_nsamples)  + '_' + str(args.seed)+ '.pt')
        else:
            profiling_mat = torch.load(args.profiling_mat_path)
            if args.bi_whitening:
                bi_whitening(args.model, model, profiling_mat, args.ratio, args.DEV)
        if not args.bi_whitening:
            whitening(args.model, model, profiling_mat, args.ratio, args.DEV)
        if args.save_path is not None:
            suffix = '_bi_whitening_only_' if args.bi_whitening else '_whitening_only_'
            model = make_model_pickleable(model).cpu()
            torch.save({'model': model, 'tokenizer': tokenizer}, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") + suffix + str(args.ratio) + '.pt')   # fp32
    elif args.step == 2:
        model, tokenizer = get_model_from_huggingface(model_id=args.model)
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
            model = make_model_pickleable(model).cpu()
            torch.save({'model': model, 'tokenizer': tokenizer}, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") +'_whitening_then_update_' + str(args.ratio) + '.pt')  # fp32
    elif args.step == 3:
        model, tokenizer = get_model_from_huggingface(args.model)
        model = model.eval()
        model = model.float()
        dataloader, _ = get_loaders(args.dataset, nsamples=args.updating_nsamples, seed=args.seed, tokenizer=tokenizer, seqlen=args.model_seq_len)
        whitening_local_update(model_name=args.model, model=model, dataloader=dataloader, profiling_mat=None, ratio=args.ratio, dev=args.DEV, direct_update=True)
        if args.save_path is not None:
            model = make_model_pickleable(model).cpu()
            torch.save({'model': model, 'tokenizer': tokenizer}, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") +'_update_only_' + str(args.ratio) + '.pt')   # fp32
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
                model = make_model_pickleable(model).cpu()
                torch.save({'model': model, 'tokenizer': tokenizer}, args.lora + '/merge.pt')
        model.eval()
        model = model.float()
        model = model.to(args.DEV)
        if args.step == 4:
            ppl_eval(model, tokenizer, datasets=['wikitext2'], model_seq_len=args.model_seq_len, batch_size=args.eval_batch_size, device=args.DEV)
        elif args.step == 5:
            eff_eval(model, tokenizer, generated_len=args.gen_seq_len, batch_size=args.eval_batch_size, device=args.DEV)
