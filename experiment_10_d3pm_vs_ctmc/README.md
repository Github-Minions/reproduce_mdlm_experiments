# 实验 10：D3PM 与 CTMC 在相同离散 Toy 任务上的对比

## 实验目的

将 D3PM 与 CTMC 放在同一离散 toy 任务下比较，控制词表、模板结构和 Transformer 规模基本一致，只改变 forward/reverse process 的时间建模方式。

## 实验设置

| 配置项 | D3PM | CTMC |
|--------|------|------|
| 时间建模 | 80 个离散时间步 | 连续时间 horizon=1.0, λ=3.0 |
| 反向步数 | 80 | 160 (tau=0.00625) |
| 序列长度 | 8, 16, 32 | 8, 16, 32 |
| 状态空间 | {A,B,C,D,[MASK]} | {A,B,C,D,[MASK]} |
| 模型 | 相同 Transformer | 相同 Transformer |

## 运行

```bash
cd experiment_10_d3pm_vs_ctmc
python run.py
```

## 预期结果

| Seq Len | Method | Token Acc | Masked Acc | Top-3 Acc | Near Rate | Runtime | NFE |
|---------|--------|-----------|------------|-----------|-----------|---------|-----|
| 8 | D3PM | 78.66% | 58.47% | 96.35% | 96.48% | 0.514s | 80 |
| 8 | CTMC | 69.17% | 54.62% | 94.70% | **98.44%** | 1.358s | 161 |
| 16 | D3PM | **85.10%** | **71.31%** | 97.75% | 88.67% | 0.833s | 80 |
| 16 | CTMC | 78.21% | 68.34% | 96.94% | **94.73%** | 1.993s | 161 |
| 32 | D3PM | **91.28%** | **82.96%** | 98.92% | 82.03% | 1.522s | 80 |
| 32 | CTMC | 88.75% | 83.42% | 98.87% | **91.21%** | 4.022s | 161 |

### 关键结论

- **D3PM** 的 token recovery accuracy 在三个长度上都高于 CTMC
- **CTMC** 的 near-template rate 在三个长度上都高于 D3PM
- CTMC 需要约 2 倍 NFE 和约 2.4-2.6 倍采样时间

## 输出文件

- `results/comparison.png` — 多维度对比图
- `results/results.json` — 数值指标汇总
