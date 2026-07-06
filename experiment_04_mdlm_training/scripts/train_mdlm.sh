#!/bin/bash
# 实验 4：MDLM 基础训练
# 需要在 kuleshov-group/mdlm 仓库根目录下运行

set -e

echo "=========================================="
echo "Experiment 4: MDLM Basic Training"
echo "=========================================="

python -u main.py \
  loader.global_batch_size=128 \
  loader.batch_size=16 \
  loader.eval_batch_size=16 \
  model=small_tinystories \
  data=tinystories \
  parameterization=subs \
  model.length=256 \
  seed=1 \
  trainer.max_steps=5000 \
  eval.compute_generative_perplexity=True \
  sampling.steps=100 \
  wandb.name=mdlm-tinystories-baseline

echo "Training complete! Checkpoints saved to outputs/"
