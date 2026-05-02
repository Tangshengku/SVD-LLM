#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
export TRANSFORMERS_CACHE=/nfs/scistore19/alistgrp/huggingface/hub

# example of compressing LLaMA-7B with SVDLLMq
FINE_TUNE_PATH="."
# run data whitening with 20% compression ratio
# python SVDLLM.py --dataset mix:wikitext2,evol-codealpaca,tulu-math --model RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic --step 1 --ratio 0.2 --whitening_nsamples 1024 --seed 3 --model_seq_len 2048 --save_path . 
## you can also run the following command for low-resource gpu (ex. llama 7b will only need 15G gpu memory to compress) or to compress large-scale llm (ex. llama 65b)
# python SVDLLM.py --model jeffwan/llama-7b-hf --step 1 --ratio 0.2 --whitening_nsamples 256 --dataset wikitext2 --model_seq_len 2048 --save_path ./ --run_low_resource
python SVDLLM.py --step 4 --model_path original --model Qwen/Qwen3-32B
# python SVDLLM.py \
#     --model mistralai/Mistral-7B-v0.1\
#     --step 6 \
#     --warm_start_ratio 0.2 \
#     --ratio 0.2 \
#     --offline_dataset mix:wikitext2,evol-codealpaca,tulu-math \
#     --on_policy_dataset mix:evol-codealpaca,tulu-math \
#     --whitening_nsamples 256 \
#     --on_policy_prompt_nsamples 256 \
#     --on_policy_prompt_len 128 \
#     --on_policy_rollout_len 512 \
#     --gradient_whitening_mode both \
#     --eval_batch_size 2 \
#     --kd_temperature 1.0 \
#     --generation_temperature 0.7 \
#     --generation_top_p 0.9 \
#     --cov_rho 0.3 \
#     --profiling_mat_path /nfs/scistore19/alistgrp/stang/SVD-LLM/mistralai_Mistral_7B_v0.1_profiling_mix:wikitext2,evol-codealpaca,tulu-math_256_3.pt \
#     --DEV cuda \
#     --save_path ./
    # --on_policy_layer_tail_ratio 0.25 \
# --profiling_mat_path /nfs/scistore19/alistgrp/stang/SVD-LLM/mistralai_Mistral_7B_v0.1_profiling_mix:wikitext2,evol-codealpaca,tulu-math_256_3.pt \

# finetune the compressed model with lora
# python utils/LoRA.py --prune_model  --data_path yahma/alpaca-cleaned --output_dir $FINE_TUNE_PATH/first_half --lora_target_modules q_u_proj,k_u_proj,v_u_proj,o_u_proj,gate_u_proj,down_u_proj,up_u_proj --lora_r 8 --num_epochs 3 --learning_rate 1e-4 --batch_size 64
# python SVDLLM.py --model_path jeffwan_llama_7b_hf_whitening_only_0.8.pt --lora $FINE_TUNE_PATH/first_half /first_half --step 4
# python utils/LoRA.py --prune_model $FINE_TUNE_PATH/first_half/merge.pt --data_path yahma/alpaca-cleaned --output_dir $FINE_TUNE_PATH/second_half --lora_target_modules q_v_proj,k_v_proj,v_v_proj,o_v_proj,gate_v_proj,down_v_proj,up_v_proj --lora_r 8 --num_epochs 3 --learning_rate 1e-4 --batch_size 64
# python SVDLLM.py --model_path jeffwan_llama_7b_hf_whitening_only_0.8.pt --lora $FINE_TUNE_PATH/first_half /first_half --step 4
# python SVDLLM.py --model_path $FINE_TUNE_PATH/first_half/merge.pt --lora $FINE_TUNE_PATH/second_half --step 4