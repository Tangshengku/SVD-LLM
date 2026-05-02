

# disable the boundary_window and tail collection
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TRANSFORMERS_CACHE=/nfs/scistore19/alistgrp/huggingface/hub/
python evo_svdllm.py \
    --model Qwen/Qwen3-32B\
    --ratio 0.2 \
    --dataset mix:wikitext2,evol-codealpaca,tulu-math \
    --source_datasets mix:wikitext2,evol-codealpaca,tulu-math \
    --whitening_nsamples 1024 \
    --search_nsamples 32 \
    --fitness_fn kl \
    --init_parent_method on_policy_gradient \
    --init_parent_on_policy_dataset mix:evol-codealpaca,tulu-math \
    --init_parent_input_covariance_source on_policy \
    --init_parent_on_policy_nsamples 512 \
    --init_parent_on_policy_layer_tail_ratio 0.25 \
    --rerank_base_fitness kl \
    --rerank_selection_fitness on_policy_plus_kl \
    --rerank_topk_on_policy 0 \
    --model_seq_len 2048 \
    --on_policy_prompt_len 128 \
    --on_policy_rollout_len 512 \
    --on_policy_eval_every 8 \
    --on_policy_temperature 1.0 \
    --rerank_on_policy_weight 0.5 \
    --generations 100 \
    --offspring 16 \
    --rank_step 128 \
    --boundary_window 32 \
    --tail_count 32 \
    --mutation_granularity group \
    --eval_batch_size 1 \
    --init_strategy uniform \
    --max_mutations 10 \
    --profiling_mat_path /nfs/scistore19/alistgrp/stang/SVD-LLM/Qwen_Qwen3_32B_profiling_mix:wikitext2,evol-codealpaca,tulu-math_1024_3.pt\
    --save_path /nfs/scistore19/alistgrp/stang/SVD-LLM/evo_output_qwen3-14b_0.2_1024whitening_rank_step128_kl_100generation_bw32_lt_0.25_512samples_rollout_512_inp_on\
    --save_model

# python evo_svdllm.py \
#     --model HUGGINGFACE_MODEL_REPO \
#     --ratio 0.2 \
#     --dataset wikitext2 \
#     --whitening_nsamples 256 \
#     --search_nsamples 16 \
#     --fitness_fn kl \
#     --generations 50 \
#     --offspring 8 \
#     --rank_step 8 \
#     --boundary_window 8 \
#     --mutation_granularity group \
#     --save_path OUTPUT_DIR \
#     --save_model