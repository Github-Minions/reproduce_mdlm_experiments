# 实验 6：AR / SEDD / MDLM 基线训练对比

## 实验目的

在统一数据集、训练步数和随机种子下，横向比较 AR、SEDD 和 MDLM 三类模型的训练行为和验证困惑度。

## 前置要求

同实验 4，需要 `kuleshov-group/mdlm` 官方代码库。

## 实验设置

| 模型 | parameterization | backbone | time_conditioning |
|------|-----------------|----------|-------------------|
| AR | `ar` | `ar` | False |
| SEDD | `sedd` | `dit` | True |
| MDLM | `subs` | `dit` | False |

其他配置统一：TinyStories, 5000 steps, seed=1, length=256。

## 运行

```bash
bash scripts/train_all_baselines.sh
```

## 预期结果

- **训练 Loss**：AR < SEDD < MDLM
- **验证 PPL**：AR 最低
- **关键结论**：训练 loss 低不代表下游修复能力强；AR 的 left-to-right 目标函数与双向 mask 修复任务不完全一致
