#!/bin/bash
# 实验 5.3：固定 step=100 的 unmask strategy 消融

set -e

CHECKPOINT="./checkpoints/mdlm_step100.ckpt"

echo "=========================================="
echo "Experiment 5.3: Unmask Strategy Ablation"
echo "=========================================="

for strategy in native random confidence_first left_to_right block; do
  echo "Evaluating unmask strategy: $strategy"
  UNMASK_STRATEGY=$strategy python -u main.py \
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

echo "Unmask strategy ablation complete!"
