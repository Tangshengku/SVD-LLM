#coding:utf8
import os
import sys
import torch
import torch.nn as nn

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)

# bandaid fix
dev = torch.device("cuda")


def _model_name(model_id):
    return model_id.lower()


def _is_opt_model(model_id):
    return "opt" in _model_name(model_id)


def _is_llama_family(model_id):
    name = _model_name(model_id)
    return "llama" in name or "vicuna" in name


def _is_mistral_family(model_id):
    name = _model_name(model_id)
    return "mistral" in name or "qwen" in name


def get_tokenizer_for_model_id(model_id, device_map="cpu"):
    from transformers import AutoTokenizer

    tokenizer_kwargs = {
        "trust_remote_code": True,
        "use_fast": False,
    }
    if device_map is not None:
        tokenizer_kwargs["device_map"] = device_map
    return AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)


def get_model_from_huggingface(model_id, device_map="auto", torch_dtype=torch.float16):
    from transformers import AutoModelForCausalLM
    tokenizer = get_tokenizer_for_model_id(model_id, device_map=device_map)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        cache_dir=None,
    )
    model.seqlen = 2048
    return model, tokenizer


def _compute_ratio_from_low_rank(low_rank, hidden_size, intermediate_size=None, is_attention=False):
    if is_attention:
        return float(2 * low_rank) / float(hidden_size)
    return float(low_rank * (intermediate_size + hidden_size)) / float(intermediate_size * hidden_size)


def _replace_linear_from_state_dict(parent_module, attr_name, weight_tensor, bias_tensor=None):
    new_linear = nn.Linear(
        weight_tensor.shape[1],
        weight_tensor.shape[0],
        bias=bias_tensor is not None,
    )
    new_linear.weight = nn.Parameter(weight_tensor.clone())
    if bias_tensor is not None:
        new_linear.bias = nn.Parameter(bias_tensor.clone())
    setattr(parent_module, attr_name, new_linear)


def _replace_attention_projections_from_state_dict(attn_module, prefix, state_dict, projection_names):
    for proj_name in projection_names:
        u_key = f"{prefix}.{proj_name}_u_proj.weight"
        v_key = f"{prefix}.{proj_name}_v_proj.weight"
        if u_key not in state_dict or v_key not in state_dict:
            continue
        bias_key = f"{prefix}.{proj_name}_u_proj.bias"
        bias_tensor = state_dict[bias_key] if bias_key in state_dict else None
        _replace_linear_from_state_dict(
            attn_module,
            f"{proj_name}_u_proj",
            state_dict[u_key],
            bias_tensor,
        )
        _replace_linear_from_state_dict(
            attn_module,
            f"{proj_name}_v_proj",
            state_dict[v_key],
        )


def _apply_svd_structure(model_id, model, state_dict):
    from component.svd_llama import SVD_LlamaAttention, SVD_LlamaMLP
    from component.svd_mistral import SVD_MistralAttention, SVD_MistralMLP
    from component.svd_opt import SVDOPTDecoderLayer

    if _is_opt_model(model_id):
        layers = model.model.decoder.layers
        layer_prefix = "model.decoder.layers"
    else:
        layers = model.model.layers
        layer_prefix = "model.layers"

    for i in range(len(layers)):
        prefix = f"{layer_prefix}.{i}"
        layer = layers[i]

        if _is_opt_model(model_id):
            q_key = f"{prefix}.self_attn.q_u_proj.weight"
            if q_key in state_dict:
                low_rank = state_dict[q_key].shape[1]
                ratio = _compute_ratio_from_low_rank(low_rank, model.config.hidden_size, is_attention=True)
                layers[i] = SVDOPTDecoderLayer(model.config, ratio=ratio)
            continue

        q_key = f"{prefix}.self_attn.q_u_proj.weight"
        if q_key in state_dict:
            low_rank = state_dict[q_key].shape[1]
            ratio = _compute_ratio_from_low_rank(low_rank, model.config.hidden_size, is_attention=True)
            if _is_llama_family(model_id):
                layer.self_attn = SVD_LlamaAttention(config=model.config, ratio=ratio)
                _replace_attention_projections_from_state_dict(
                    layer.self_attn,
                    f"{prefix}.self_attn",
                    state_dict,
                    ["q", "k", "v", "o"],
                )
            else:
                layer.self_attn = SVD_MistralAttention(config=model.config, ratio=ratio)
                _replace_attention_projections_from_state_dict(
                    layer.self_attn,
                    f"{prefix}.self_attn",
                    state_dict,
                    ["q", "k", "v", "o"],
                )

        mlp_key = f"{prefix}.mlp.gate_u_proj.weight"
        if mlp_key in state_dict:
            low_rank = state_dict[mlp_key].shape[1]
            ratio = _compute_ratio_from_low_rank(
                low_rank,
                model.config.hidden_size,
                model.config.intermediate_size,
                is_attention=False,
            )
            if _is_llama_family(model_id):
                layer.mlp = SVD_LlamaMLP(
                    hidden_size=layer.hidden_size,
                    intermediate_size=model.config.intermediate_size,
                    hidden_act=model.config.hidden_act,
                    ratio=ratio,
                )
            else:
                layer.mlp = SVD_MistralMLP(config=model.config, ratio=ratio)

    return model


def _infer_base_model_id(saved_model, tokenizer, base_model_id=None):
    if base_model_id is not None:
        return base_model_id
    if hasattr(saved_model, "config") and getattr(saved_model.config, "_name_or_path", None):
        return saved_model.config._name_or_path
    if getattr(tokenizer, "name_or_path", None):
        return tokenizer.name_or_path
    raise ValueError("Could not infer the original Hugging Face model id from the local checkpoint. Please pass `base_model_id`.")


def _checkpoint_metadata_from_loaded(loaded, base_model_id=None):
    inferred_base_model_id = loaded.get("base_model_id", None)
    if base_model_id is not None:
        inferred_base_model_id = base_model_id
    tokenizer_name_or_path = loaded.get("tokenizer_name_or_path", inferred_base_model_id)
    seqlen = loaded.get("seqlen", 2048)
    return inferred_base_model_id, tokenizer_name_or_path, seqlen


def load_checkpoint_payload(model_id, base_model_id=None):
    loaded = torch.load(model_id, weights_only=False, map_location="cpu")
    if "model_state_dict" in loaded:
        inferred_base_model_id, tokenizer_name_or_path, seqlen = _checkpoint_metadata_from_loaded(
            loaded,
            base_model_id=base_model_id,
        )
        tokenizer = get_tokenizer_for_model_id(tokenizer_name_or_path)
        return {
            "format": "state_dict",
            "state_dict": loaded["model_state_dict"],
            "tokenizer": tokenizer,
            "base_model_id": inferred_base_model_id,
            "tokenizer_name_or_path": tokenizer_name_or_path,
            "seqlen": seqlen,
        }

    tokenizer, model = loaded["tokenizer"], loaded["model"]
    inferred_base_model_id = _infer_base_model_id(model, tokenizer, base_model_id=base_model_id)
    return {
        "format": "pickled_model",
        "model": model,
        "state_dict": model.state_dict(),
        "tokenizer": tokenizer,
        "base_model_id": inferred_base_model_id,
        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", inferred_base_model_id),
        "seqlen": getattr(model, "seqlen", 2048),
    }


def save_compressed_checkpoint(path, model, tokenizer, base_model_id):
    checkpoint = {
        "checkpoint_format": "svdllm_state_dict_v2",
        "base_model_id": base_model_id,
        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", base_model_id),
        "seqlen": getattr(model, "seqlen", 2048),
        "model_state_dict": model.state_dict(),
    }
    torch.save(checkpoint, path)


def get_model_from_local(model_id, base_model_id=None, reconstruct=False, device_map="cpu", torch_dtype=torch.float16):
    payload = load_checkpoint_payload(model_id, base_model_id=base_model_id)
    tokenizer = payload["tokenizer"]

    if payload["format"] == "pickled_model" and not reconstruct:
        return payload["model"], tokenizer

    model_id = payload["base_model_id"]
    rebuilt_model, _ = get_model_from_huggingface(model_id, device_map=device_map, torch_dtype=torch_dtype)
    rebuilt_model = _apply_svd_structure(model_id, rebuilt_model, payload["state_dict"])
    load_result = rebuilt_model.load_state_dict(payload["state_dict"], strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        print("Warning: non-strict state_dict load when reconstructing local model.")
        print("Missing keys:", load_result.missing_keys[:20])
        print("Unexpected keys:", load_result.unexpected_keys[:20])
    rebuilt_model.seqlen = payload["seqlen"]
    return rebuilt_model, tokenizer


def _set_module_tensor(module, name, tensor):
    current = getattr(module, name)
    current.data = tensor.to(dtype=current.data.dtype, device=current.data.device)


def load_dense_model_from_compressed_checkpoint(checkpoint_path, base_model_id=None, device_map="cpu", torch_dtype=torch.float16):
    payload = load_checkpoint_payload(checkpoint_path, base_model_id=base_model_id)
    model_id = payload["base_model_id"]
    state_dict = payload["state_dict"]
    model, tokenizer = get_model_from_huggingface(model_id, device_map=device_map, torch_dtype=torch_dtype)

    direct_state = {}
    base_state = model.state_dict()
    for key, tensor in state_dict.items():
        if key in base_state and base_state[key].shape == tensor.shape:
            direct_state[key] = tensor
    model.load_state_dict(direct_state, strict=True)

    if _is_opt_model(model_id):
        layers = model.model.decoder.layers
        layer_prefix = "model.decoder.layers"
    else:
        layers = model.model.layers
        layer_prefix = "model.layers"

    for i in range(len(layers)):
        prefix = f"{layer_prefix}.{i}"
        layer = layers[i]

        if _is_opt_model(model_id):
            specs = [
                ("self_attn.q_proj", "self_attn.q_u_proj.weight", "self_attn.q_v_proj.weight", "self_attn.q_u_proj.bias"),
                ("self_attn.k_proj", "self_attn.k_u_proj.weight", "self_attn.k_v_proj.weight", "self_attn.k_u_proj.bias"),
                ("self_attn.v_proj", "self_attn.v_u_proj.weight", "self_attn.v_v_proj.weight", "self_attn.v_u_proj.bias"),
                ("self_attn.out_proj", "self_attn.out_u_proj.weight", "self_attn.out_v_proj.weight", "self_attn.out_u_proj.bias"),
                ("fc1", "fc1_u_proj.weight", "fc1_v_proj.weight", "fc1_u_proj.bias"),
                ("fc2", "fc2_u_proj.weight", "fc2_v_proj.weight", "fc2_u_proj.bias"),
            ]
            for target_name, u_name, v_name, bias_name in specs:
                u_key = f"{prefix}.{u_name}"
                v_key = f"{prefix}.{v_name}"
                if u_key not in state_dict:
                    continue
                target_module = layer
                for attr in target_name.split("."):
                    target_module = getattr(target_module, attr)
                dense_weight = state_dict[u_key].float() @ state_dict[v_key].float()
                _set_module_tensor(target_module, "weight", dense_weight)
                bias_key = f"{prefix}.{bias_name}"
                if bias_key in state_dict and target_module.bias is not None:
                    _set_module_tensor(target_module, "bias", state_dict[bias_key].float())
            continue

        specs = [
            ("self_attn.q_proj", "self_attn.q_u_proj.weight", "self_attn.q_v_proj.weight"),
            ("self_attn.k_proj", "self_attn.k_u_proj.weight", "self_attn.k_v_proj.weight"),
            ("self_attn.v_proj", "self_attn.v_u_proj.weight", "self_attn.v_v_proj.weight"),
            ("self_attn.o_proj", "self_attn.o_u_proj.weight", "self_attn.o_v_proj.weight"),
            ("mlp.gate_proj", "mlp.gate_u_proj.weight", "mlp.gate_v_proj.weight"),
            ("mlp.down_proj", "mlp.down_u_proj.weight", "mlp.down_v_proj.weight"),
            ("mlp.up_proj", "mlp.up_u_proj.weight", "mlp.up_v_proj.weight"),
        ]
        for target_name, u_name, v_name in specs:
            u_key = f"{prefix}.{u_name}"
            v_key = f"{prefix}.{v_name}"
            if u_key not in state_dict:
                continue
            target_module = layer
            for attr in target_name.split("."):
                target_module = getattr(target_module, attr)
            dense_weight = state_dict[u_key].float() @ state_dict[v_key].float()
            _set_module_tensor(target_module, "weight", dense_weight)

    model.seqlen = payload["seqlen"]
    return model, tokenizer


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res
