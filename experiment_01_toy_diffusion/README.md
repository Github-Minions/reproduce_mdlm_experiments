# 实验 1：连续二维 Toy Diffusion 复现

## 实验目的

用二维分布直观验证 DDPM/score-based diffusion 的基本流程：
- **前向过程**：逐步把结构化数据破坏成近似噪声
- **反向过程**：从噪声中恢复目标分布的多模态或流形形状

## 数据集

| 数据集 | 特点 | 描述 |
|--------|------|------|
| two_moons | 双月形，非凸 | sklearn `make_moons` |
| swiss_roll | 瑞士卷流形 | sklearn `make_s_curve` 取两维 |
| gaussian_mixture | 8组分圆环排列 | 8个高斯分布在圆周上 |

## 实验配置

| 配置项 | 值 |
|--------|-----|
| 每个数据集训练步数 | 20,000 |
| 扩散时间步 | 1,000 |
| 采样生成样本数 | 2,048 |
| beta schedule | linear (1e-4 -> 0.02) |
| Score network | MLP (2+1 -> 128 -> 128 -> 128 -> 2) |
| 优化器 | Adam, lr=1e-3 |
| batch size | 256 |

## 评估指标

| 指标 | 说明 |
|------|------|
| MMD-RBF | 最大均值差异（RBF核） |
| Sliced Wasserstein | 随机投影近似Wasserstein距离 |
| Histogram KL | 二维直方图KL散度 |
| Mean error | 均值误差 |
| Cov error | 协方差误差 |

## 运行

```bash
cd experiment_01_toy_diffusion
python run.py
```

## 预期结果

| 数据集 | MMD-RBF | Sliced W. | Mean Err | Cov Err |
|--------|---------|-----------|----------|---------|
| two_moons | ~0.00075 | ~3.0 | ~0.024 | ~0.069 |
| swiss_roll | ~0.00039 | ~3.7 | ~0.031 | ~0.023 |
| gaussian_mixture | ~0.00014 | ~2.6 | ~0.030 | ~0.016 |

## 输出文件

运行后 `results/` 目录下会生成：
- `forward_noise.png` — 前向加噪过程可视化
- `loss_curves.png` — 训练损失曲线
- `generated_vs_real.png` — 真实样本与生成样本对比
- `results.json` — 数值指标汇总
