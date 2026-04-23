

# disable the boundary_window and tail collection
export CUDA_VISIBLE_DEVICES=0

python evo_svdllm.py \
    --model mistralai/Mistral-7B-v0.1 \
    --ratio 0.4 \
    --dataset mix:wikitext2,evol-codealpaca,tulu-math \
    --source_datasets wikitext2,evol-codealpaca,tulu-math \
    --whitening_nsamples 256 \
    --search_nsamples 32 \
    --fitness_fn kl \
    --rerank_base_fitness kl \
    --rerank_topk_on_policy 4 \
    --model_seq_len 2048 \
    --on_policy_prompt_len 128 \
    --on_policy_rollout_len 64 \
    --on_policy_eval_every 8 \
    --on_policy_temperature 1.0 \
    --generations 50 \
    --offspring 16 \
    --rank_step 128 \
    --boundary_window 32 \
    --tail_count 32 \
    --mutation_granularity group \
    --eval_batch_size 16 \
    --init_strategy uniform \
    --max_mutations 5 \
    --save_path /nfs/scistore19/alistgrp/stang/SVD-LLM/evo_output_mistral_rank_step128_kl_50generation_bw32_mixture_search_onpolicy_kl \
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