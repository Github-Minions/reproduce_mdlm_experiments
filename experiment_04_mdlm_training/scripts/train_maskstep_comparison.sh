#!/bin/bash
# 实验 4.2：不同训练 mask-step 的对比

set -e

echo "=========================================="
echo "Experiment 4.2: Mask-Step Comparison"
echo "=========================================="

for steps in 10 50 100 200; do
  echo "Training with mask-step=$steps"
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
    sampling.steps=$steps \
    wandb.name=mdlm-tinystories-maskstep$steps
done

echo "Mask-step comparison complete!"
