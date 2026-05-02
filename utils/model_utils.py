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

def get_model_from_huggingface(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    max_memory=None,
    attn_implementation="flash_attention_2",
):
    from transformers import AutoModelForCausalLM, LlamaTokenizer, AutoTokenizer, LlamaForCausalLM
    model_id_lower = model_id.lower()
    if "opt" in model_id_lower or "mistral" in model_id_lower or "qwen" in model_id_lower or "llama" in model_id_lower:
        tokenizer = AutoTokenizer.from_pretrained(model_id, device_map="cpu", trust_remote_code=True, use_fast=False)
    else:
        tokenizer = LlamaTokenizer.from_pretrained(model_id, device_map="cpu", trust_remote_code=True)
    load_kwargs = dict(
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        cache_dir=None,
    )
    if device_map is not None and str(device_map).lower() != "none":
        load_kwargs["device_map"] = device_map
    if max_memory is not None:
        load_kwargs["max_memory"] = max_memory
    if "opt" not in model_id_lower and attn_implementation:
        load_kwargs["attn_implementation"] = attn_implementation
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    except TypeError:
        load_kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.seqlen = 2048
    return model, tokenizer

def get_model_from_local(model_id):
    pruned_dict = torch.load(model_id, weights_only=False, map_location='cpu')
    tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model']
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
