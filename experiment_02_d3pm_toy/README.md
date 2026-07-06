# 实验 2：D3PM 离散状态 Toy 实验

## 实验目的

将扩散过程从连续二维空间转到离散 token 空间，比较 D3PM 中三类 forward corruption（uniform/absorbing/structured）对 denoising 难度、token 恢复准确率和反向采样质量的影响。

## 实验设置

| 配置项 | 值 |
|--------|-----|
| 状态空间 | {A, B, C, D, [MASK]} |
| 序列长度 | 16 |
| 扩散步数 | 100 |
| Corruption 类型 | uniform / absorbing / structured |
| 训练步数 | 10,000 |
| Batch size | 256 |
| 模型 | Transformer Encoder (d=64, 4 heads, 3 layers) |

## 三种 Q 矩阵

| 类型 | 设计 |
|------|------|
| **uniform** | 所有转移等概率 |
| **absorbing** | 所有状态转移到 [MASK] |
| **structured** | A<->B, C<->D 有较高转移概率，70%自环，10%到mask |

## 评估指标

- Validation Cross-Entropy
- Token Recovery Accuracy
- Top-3 Accuracy
- Exact Template Rate
- Near Template Rate

## 运行

```bash
cd experiment_02_d3pm_toy
python run.py
```

## 预期结果

| Corruption | Val. CE | Token Acc | Top-3 Acc |
|------------|---------|-----------|-----------|
| uniform | 0.659 | 70.52% | 94.03% |
| absorbing | 0.411 | 80.22% | 96.83% |
| **structured** | **0.328** | **85.62%** | **99.01%** |

## 输出文件

- `results/loss_*.png` — 训练损失曲线
- `results/Q_heatmap_*.png` — 转移矩阵热力图
- `results/results.json` — 数值指标汇总
