#!/bin/bash
# 实验 4.3：不同噪声日程的训练稳定性对比

set -e

echo "=========================================="
echo "Experiment 4.3: Noise Schedule Comparison"
echo "=========================================="

for schedule in linear baseline loglinear; do
  echo "Training with noise schedule=$schedule"
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
    noise=$schedule \
    sampling.steps=100 \
    wandb.name=mdlm-tinystories-noise-$schedule
done

echo "Noise schedule comparison complete!"
