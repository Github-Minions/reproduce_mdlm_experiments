#!/bin/bash
# 实验 6：AR / SEDD / MDLM 基线对比训练

set -e

echo "=========================================="
echo "Experiment 6: Baseline Comparison"
echo "=========================================="

# AR
python -u main.py \
  model=small_tinystories \
  data=tinystories \
  parameterization=ar \
  backbone=ar \
  model.length=256 \
  seed=1 \
  trainer.max_steps=5000 \
  wandb.name=ar-tinystories

# SEDD
python -u main.py \
  model=small_tinystories \
  data=tinystories \
  parameterization=sedd \
  backbone=dit \
  model.length=256 \
  time_conditioning=True \
  sampling.predictor=analytic \
  seed=1 \
  trainer.max_steps=5000 \
  wandb.name=sedd-tinystories

# MDLM (reference)
python -u main.py \
  model=small_tinystories \
  data=tinystories \
  parameterization=subs \
  model.length=256 \
  seed=1 \
  trainer.max_steps=5000 \
  wandb.name=mdlm-tinystories-ref

echo "All baselines trained!"
