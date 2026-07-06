# 实验 3：CTMC Discrete Diffusion 与 Tau-Leaping

## 实验目的

将离散扩散从固定离散时间步推广到连续时间 Markov Chain，验证：
1. CTMC 闭式 mask ratio 与仿真结果是否一致
2. tau-leaping 的误差-效率权衡

## 实验设置

| 配置项 | 值 |
|--------|-----|
| Forward process | Absorbing-mask CTMC |
| 时间区间 | [0, 1] |
| Mask rate (λ) | 3.0 |
| 序列长度 | 16 |
| 状态空间 | {A, B, C, D, [MASK]} |
| Tau-leaping τ | 0.1, 0.05, 0.02, 0.01, 0.005 |
| 训练步数 | 10,000 |

## 核心公式

**闭式 mask ratio**：
```
r(t) = 1 - exp(-λt)
```

**Rate matrix**：
```
R(x, y) = λ  if x ≠ mask, y = mask
          0  otherwise
```

## 运行

```bash
cd experiment_03_ctmc_toy
python run.py
```

## 预期结果

| 指标 | 值 |
|------|-----|
| Validation CE | ~0.450 |
| Token recovery accuracy | ~78.59% |
| Masked token recovery | ~68.90% |
| Top-3 accuracy | ~97.13% |
| Near-template rate (Hamming ≤2) | ~99.02% |

### Tau-Leaping 误差-效率权衡

| τ | TV Distance | Runtime | NFE |
|---|-------------|---------|-----|
| 0.100 | 0.1923 | 0.020s | 10 |
| 0.050 | 0.0894 | 0.045s | 20 |
| 0.020 | 0.0345 | 0.116s | 50 |
| 0.010 | 0.0222 | 0.227s | 100 |
| 0.005 | 0.0111 | 0.412s | 200 |

## 输出文件

- `results/mask_ratio_check.png` — 闭式 mask ratio 校验
- `results/loss.png` — 训练损失曲线
- `results/recovery_by_time.png` — 按时间的恢复难度
- `results/tau_ablation.png` — tau-leaping 误差-时间折中
- `results/results.json` — 数值指标汇总
