# 实验 8：Token 类型分组准确率

## 实验目的

分析不同 token 类型（English / Punctuation / Rare）上的恢复误差，揭示模型在不同词类上的修复能力差异。

## 前置要求

基于实验 7 的修复结果，按 token 类型分组统计。

## Token 分类标准

| 类型 | 说明 | 示例 |
|------|------|------|
| English | 常见英文单词 | "the", "cat", "happy" |
| Punctuation | 标点符号 | ",", ".", "!", '"' |
| Rare | 低频词 / 专有名词 | "Zyx", "xylophone" |

## 运行

```bash
cd experiment_08_token_type
python run.py --repair_results ../experiment_07_mask_repair/results/results.json
```

或直接从实验 7 的结果中分析：
```python
from run import analyze_by_token_type
results = analyze_by_token_type(val_texts, repaired_texts, mask_positions)
```

## 预期结果

| 任务 | 模型 | English | Punctuation | Rare |
|------|------|---------|-------------|------|
| Random 15% | AR | 55.24% | 75.60% | 41.67% |
| Random 15% | SEDD | 76.93% | 92.37% | 55.56% |
| Random 15% | **MDLM** | **78.91%** | **94.17%** | **80.77%** |
| OCR-like 15% | AR | 59.83% | 78.02% | 40.91% |
| OCR-like 15% | SEDD | 78.04% | 93.79% | 72.00% |
| OCR-like 15% | **MDLM** | **79.63%** | **89.25%** | **67.86%** |
| Span len=10 | AR | 24.91% | 43.86% | 14.29% |
| Span len=10 | SEDD | 28.64% | 56.82% | 0.00% |
| Span len=10 | **MDLM** | **33.10%** | **55.07%** | **0.00%** |

## 关键发现

- **Punctuation** 准确率通常高于 English（局部句法结构可预测）
- **Rare** token 样本数少，统计不具普遍性
- Span Mask 场景下所有类型准确率均显著下降
