#!/bin/bash
# 实验 5.2：固定 step=100 的 token selection strategy 消融

set -e

CHECKPOINT="./checkpoints/mdlm_step100.ckpt"

echo "=========================================="
echo "Experiment 5.2: Token Selection Ablation"
echo "=========================================="

for strategy in greedy categorical_topk7 categorical_topk10 categorical_topk13 topk5_temp1 topk10_temp1 topk20_temp1; do
  echo "Evaluating token selection: $strategy"
  SAMPLER_STRATEGY=$strategy python -u main.py \
    mode=sample_eval \
    eval.checkpoint_path=$CHECKPOINT \
    data=tinystories \
    model=small_tinystories \
    model.length=256 \
    sampling.predictor=ddpm_cache \
    sampling.steps=100 \
    loader.eval_batch_size=16 \
    sampling.num_sample_batches=10 \
    eval.compute_generative_perplexity=True \
    eval.gen_ppl_eval_model_name_or_path=gpt2
done

echo "Token selection ablation complete!"
