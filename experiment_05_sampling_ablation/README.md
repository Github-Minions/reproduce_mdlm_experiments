# 实验 5：采样步数与固定步数策略消融

## 实验目的

固定已训练的 MDLM checkpoint，只改变推理阶段策略，研究采样步数、token selection 和 unmask 策略对生成质量和效率的影响。

## 前置要求

需要实验 4 训练好的 MDLM checkpoint，或下载预训练权重：

```python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="goosemaths/tinystories_masked_diffusion_model_ckpt",
                filename="mdlm_step100.ckpt", local_dir="checkpoints")
```

## 子实验

### 5.1 采样步数与质量-效率折中

测试 10/20/50/100/200/500/1000 步的生成 PPL 和运行时间。

### 5.2 Token Selection Strategy

固定 step=100，比较：
- greedy
- categorical (temperature=0.7, 1.0, 1.3)
- top-k (k=5, 10, 20, temperature=1.0)

### 5.3 Unmask Strategy

固定 step=100，比较：
- native（原论文默认）
- random
- confidence-first
- left-to-right
- block

## 运行

```bash
# Step 1: 复制自定义 sampler 到官方代码
mkdir -p /path/to/mdlm/custom_sampling
cp custom_sampler.py /path/to/mdlm/custom_sampling/

# Step 2: 运行各子实验
cd /path/to/mdlm
bash /path/to/reproduce_mdlm_experiments/experiment_05/scripts/ablation_steps.sh
bash /path/to/reproduce_mdlm_experiments/experiment_05/scripts/ablation_token_selection.sh
bash /path/to/reproduce_mdlm_experiments/experiment_05/scripts/ablation_unmask.sh
```

## 预期结果

### 采样步数消融

| Steps | Gen PPL | Runtime |
|-------|---------|---------|
| 10 | 87.02 | 10.89s |
| 20 | 51.74 | 16.13s |
| 50 | 41.77 | 14.61s |
| **100** | **30.98** | **19.41s** |
| 200 | 31.15 | 29.90s |
| 500 | 28.99 | 62.24s |
| 1000 | 28.45 | 115.68s |

### Token Selection (step=100)

| Strategy | Gen PPL |
|----------|---------|
| greedy | 42.43 |
| categorical/temp13 | 653.44 (退化) |
| **topK5/temp1** | **11.81** (最优) |

### Unmask Strategy (step=100)

| Strategy | Gen PPL |
|----------|---------|
| native | 39.67 |
| confidence-first | 7.90 |
| left-to-right | 2.19 |
| block | 2.19 |

> 注意：left-to-right 和 block 的 PPL 极低但可能伴随退化模式。
