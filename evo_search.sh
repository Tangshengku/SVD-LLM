

# disable the boundary_window and tail collection
 python evo_svdllm.py \
    --model HUGGINGFACE_MODEL_REPO \
    --ratio 0.2 \
    --dataset wikitext2 \
    --whitening_nsamples 256 \
    --search_nsamples 16 \
    --model_seq_len 2048 \
    --fitness_fn kl \
    --generations 50 \
    --offspring 8 \
    --rank_step 8 \
    --boundary_window 0 \
    --tail_count 0 \
    --mutation_granularity group \
    --save_path OUTPUT_DIR \
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