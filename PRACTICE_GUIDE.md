# Diffusion & MDLM 实验复现实践指导手册

> 本手册面向希望复现 DDPM3 报告中 10 个实验的研究者，提供从环境搭建到结果分析的完整操作指南。

---

## 目录

1. [环境准备](#1-环境准备)
2. [数据获取](#2-数据获取)
3. [Phase 1：Toy 实验快速启动](#3-phase-1toy-实验快速启动)
4. [Phase 2：MDLM 模型训练](#4-phase-2mdlm-模型训练)
5. [Phase 3：采样策略消融](#5-phase-3采样策略消融)
6. [Phase 4：下游修复评估](#6-phase-4下游修复评估)
7. [常见问题排查](#7-常见问题排查)
8. [结果分析指南](#8-结果分析指南)

---

## 1. 环境准备

### 1.1 克隆本仓库

```bash
git clone https://github.com/Github-Minions/reproduce_mdlm_experiments.git
cd reproduce_mdlm_experiments
```

### 1.2 创建 Python 环境

```bash
# 推荐使用 conda
conda create -n mdlm_repro python=3.10
conda activate mdlm_repro

# 安装基础依赖
pip install -r requirements.txt
```

**依赖清单说明**：

| 包名 | 用途 | 哪些实验需要 |
|------|------|-------------|
| torch | 深度学习框架 | 全部 |
| numpy | 数值计算 | 全部 |
| matplotlib | 可视化 | 全部 |
| scipy | 科学计算 | 实验 1 |
| scikit-learn | 数据集生成 | 实验 1 |
| datasets | HuggingFace 数据集 | 数据准备 |
| transformers | Tokenizer | 实验 7-9 |
| editdistance | 编辑距离 | 实验 7 |

### 1.3 为实验 4-6 准备 MDLM 官方代码

实验 1-3, 7-10 的代码已包含在本仓库中，可直接运行。实验 4-6 需要官方代码库：

```bash
# 在仓库同级目录克隆
cd ..
git clone https://github.com/kuleshov-group/mdlm.git
cd mdlm
conda env create -f requirements.yaml
conda activate mdlm
```

---

## 2. 数据获取

### 2.1 自动下载 TinyStories

```bash
cd reproduce_mdlm_experiments/data
python prepare_data.py
```

首次运行会自动从 HuggingFace 下载 `roneneldan/TinyStories`，下载后：
- 训练集：2,119,719 条
- 验证集：21,990 条
- 固定 100 条验证样本：`data/val_texts_100.pt`

### 2.2 下载预训练权重（跳过训练时使用）

```python
from huggingface_hub import hf_hub_download

repo_id = "goosemaths/tinystories_masked_diffusion_model_ckpt"

# MDLM step=100
hf_hub_download(repo_id=repo_id, filename="mdlm_step100.ckpt",
                local_dir="checkpoints")

# SEDD
hf_hub_download(repo_id=repo_id, filename="sedd.ckpt",
                local_dir="checkpoints")
```

---

## 3. Phase 1：Toy 实验快速启动

Toy 实验（1, 2, 3, 10）**完全独立**，不依赖 MDLM 官方代码，安装好 `requirements.txt` 后即可运行。

### 3.1 实验 1：连续二维 Diffusion

```bash
cd experiment_01_toy_diffusion
python run.py
```

**预期输出**：
- `results/forward_*.png` — 前向加噪过程可视化
- `results/gen_*.png` — 真实 vs 生成样本对比
- `results/results.json` — MMD、Sliced Wasserstein 等指标

**运行时间**：CPU 约 10-15 分钟，GPU 约 3-5 分钟

### 3.2 实验 2：D3PM 离散 Toy

```bash
cd experiment_02_d3pm_toy
python run.py
```

**预期输出**：
- `results/Q_heatmap_*.png` — 三种 Q 矩阵热力图
- `results/results.json` — 三种 corruption 的准确率对比

**关键观察**：structured corruption 的 token recovery 最高（~85%），uniform 最低（~70%）

### 3.3 实验 3：CTMC 与 Tau-Leaping

```bash
cd experiment_03_ctmc_toy
python run.py
```

**预期输出**：
- `results/mask_ratio_check.png` — 闭式 mask ratio 验证
- `results/tau_ablation.png` — tau 与误差/时间的权衡曲线

**关键观察**：tau=0.01 时 TV distance 约 0.022，是精度和效率的较好平衡

### 3.4 实验 10：D3PM vs CTMC 对比

```bash
cd experiment_10_d3pm_vs_ctmc
python run.py
```

**预期输出**：
- `results/comparison.png` — 多维度对比图

**运行时间**：约 20-30 分钟（训练 6 个模型）

---

## 4. Phase 2：MDLM 模型训练

### 4.1 复制配置文件

```bash
cd /path/to/mdlm

# 复制 TinyStories 配置
cp /path/to/reproduce_mdlm_experiments/experiment_04/configs/tinystories.yaml \
   configs/data/

cp /path/to/reproduce_mdlm_experiments/experiment_04/configs/small_tinystories.yaml \
   configs/model/
```

### 4.2 训练 MDLM（实验 4）

```bash
cd /path/to/mdlm
bash /path/to/reproduce_mdlm_experiments/experiment_04/scripts/train_mdlm.sh
```

**预期行为**：
- 训练 loss 整体下降
- 验证 PPL 早期可能有尖峰，随后回落
- 最终 PPL 约在 30-40 之间

**运行时间**：单卡 A100 约 2-3 小时

### 4.3 训练基线对比（实验 6）

```bash
cd /path/to/mdlm
bash /path/to/reproduce_mdlm_experiments/experiment_06/scripts/train_all_baselines.sh
```

这会依次训练 AR、SEDD、MDLM 三个模型。

**运行时间**：单卡 A100 约 6-8 小时（三个模型）

---

## 5. Phase 3：采样策略消融

### 5.1 准备

确保已有 `checkpoints/mdlm_step100.ckpt`（实验 4 训练产出或预训练权重）。

### 5.2 运行采样步数消融（实验 5.1）

```bash
cd /path/to/mdlm
bash /path/to/reproduce_mdlm_experiments/experiment_05/scripts/ablation_steps.sh
```

### 5.3 运行 Token Selection 消融（实验 5.2）

```bash
cd /path/to/mdlm
bash /path/to/reproduce_mdlm_experiments/experiment_05/scripts/ablation_token_selection.sh
```

**注意**：需要在 `diffusion.py` 的采样循环中集成 `custom_sampler.py` 的策略函数。

### 5.4 运行 Unmask 策略消融（实验 5.3）

```bash
cd /path/to/mdlm
bash /path/to/reproduce_mdlm_experiments/experiment_05/scripts/ablation_unmask.sh
```

---

## 6. Phase 4：下游修复评估

### 6.1 实验 7：Mask 修复

```bash
cd reproduce_mdlm_experiments/experiment_07_mask_repair
python run.py
```

**注意**：当前 `run.py` 使用占位修复函数。实际使用时需要加载真实 checkpoint 并替换 `dummy_repair` 函数。

### 6.2 实验 8：Token 类型分组

```bash
cd reproduce_mdlm_experiments/experiment_08_token_type
python run.py
```

### 6.3 实验 9：Top-K 交互式修复

```bash
cd reproduce_mdlm_experiments/experiment_09_topk_repair
python run.py
```

---

## 7. 常见问题排查

### 问题 1：TinyStories 下载失败

**现象**：`ConnectionError` 或超时

**解决**：
```bash
# 使用镜像
export HF_ENDPOINT=https://hf-mirror.com
python data/prepare_data.py
```

### 问题 2：CUDA OOM

**现象**：`RuntimeError: CUDA out of memory`

**解决**：
- 减小 batch size：`loader.batch_size=8`
- 使用更小的模型：`model.length=128`
- 或使用 CPU 运行（toy 实验支持 CPU）

### 问题 3：缺少 tokenizer

**现象**：`OSError: Can't load tokenizer`

**解决**：
```python
from transformers import GPT2Tokenizer
# 首次使用会自动下载
```

### 问题 4：实验 4-6 找不到配置

**现象**：`Could not load 'configs/data/tinystories.yaml'`

**解决**：确保已将配置文件复制到官方 mdlm 仓库的 configs 目录

---

## 8. 结果分析指南

### 如何判断实验是否成功？

| 实验 | 成功标志 | 失败标志 |
|------|----------|----------|
| 实验 1 | MMD < 0.001 | MMD > 0.01 |
| 实验 2 | structured > absorbing > uniform | 顺序颠倒 |
| 实验 3 | 闭式与仿真 mask ratio 重合 | 偏差 > 5% |
| 实验 4 | Loss 下降，PPL < 50 | Loss 发散 |
| 实验 5 | 100 步 PPL 最优 | 1000 步更差 |
| 实验 6 | AR loss < SEDD < MDLM | 顺序颠倒 |
| 实验 7 | MDLM > SEDD > AR | 顺序颠倒 |
| 实验 8 | Punctuation > English | Rare 最高 |
| 实验 9 | Top-3 > Top-1 + 10pp | 提升 < 5pp |
| 实验 10 | D3PM acc > CTMC | CTMC 全面领先 |

### 关键指标速查

| 指标 | 含义 | 越高越好？ |
|------|------|-----------|
| Mask Accuracy | mask 位置正确恢复率 | 是 |
| EDR | 编辑距离缩减率 | 是 |
| Gen PPL | 生成文本困惑度 | 否（越低越好）|
| TV Distance | 真实/生成分布差异 | 否 |
| Near Rate | 模板匹配率 | 是 |

---

*本手册与 reproduce_mdlm_experiments 仓库同步更新。如有问题请提交 Issue。*
