# Diffusion & MDLM 实验复现实践指导手册

> 本手册面向希望复现 DDPM3 报告中 10 个实验的研究者，提供从环境搭建到结果分析的完整操作指南。

---

## 目录

1. [环境准备](#1-环境准备)
2. [项目结构与运行前后变化](#2-项目结构与运行前后变化)
3. [数据获取](#3-数据获取)
4. [Phase 1：Toy 实验快速启动](#4-phase-1toy-实验快速启动)
5. [Phase 2：MDLM 模型训练](#5-phase-2mdlm-模型训练)
6. [Phase 3：采样策略消融](#6-phase-3采样策略消融)
7. [Phase 4：下游修复评估](#7-phase-4下游修复评估)
8. [常见问题排查](#8-常见问题排查)
9. [结果分析指南](#9-结果分析指南)

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

## 2. 项目结构与运行前后变化

### 2.1 克隆后（运行前）的目录结构

```
reproduce_mdlm_experiments/
├── README.md                          # 项目说明
├── PRACTICE_GUIDE.md                  # 本手册
├── requirements.txt                   # Python 依赖
├── data/
│   └── prepare_data.py               # 数据准备脚本
│
├── experiment_01_toy_diffusion/
│   ├── README.md                     # 实验说明
│   └── run.py                        # 实验代码
│   # └── results/                    # [运行后自动创建] 结果目录
│
├── experiment_02_d3pm_toy/
│   ├── README.md
│   └── run.py
│   # └── results/
│
├── experiment_03_ctmc_toy/
│   ├── README.md
│   └── run.py
│   # └── results/
│
├── experiment_04_mdlm_training/
│   ├── README.md
│   ├── configs/                      # YAML 配置文件
│   └── scripts/                      # 训练脚本
│   # └── results/                    # [运行后创建]
│
├── experiment_05_sampling_ablation/
│   ├── README.md
│   ├── custom_sampler.py
│   └── scripts/                      # 消融脚本
│   # └── results/
│
├── experiment_06_baseline_comparison/
│   ├── README.md
│   └── scripts/
│   # └── results/
│
├── experiment_07_mask_repair/
│   ├── README.md
│   └── run.py
│   # └── results/
│
├── experiment_08_token_type/
│   ├── README.md
│   └── run.py
│   # └── results/
│
├── experiment_09_topk_repair/
│   ├── README.md
│   └── run.py
│   # └── results/
│
└── experiment_10_d3pm_vs_ctmc/
    ├── README.md
    └── run.py
    # └── results/
```

> `#` 注释的行表示**运行后才会出现**，克隆时不存在。

### 2.2 数据准备后的变化

```bash
cd data && python prepare_data.py
```

运行后新增：
```
data/
├── prepare_data.py                    # 原文件
├── cache/                            # [新增] HuggingFace 数据集缓存
│   └── ...
├── val_texts_100.pt                  # [新增] 100条验证文本
└── val_set_100.pt                    # [新增] 100条验证 token IDs
```

| 文件 | 内容 | 下游用途 |
|------|------|----------|
| `val_texts_100.pt` | `list[str]` 原始文本 | 实验 7-9 的 mask 修复输入 |
| `val_set_100.pt` | `list[list[int]]` token IDs | 需要 tokenizer 编码时使用 |
| `cache/` | HuggingFace 数据集原始文件 | 首次下载后复用，无需重复下载 |

### 2.3 实验运行后的目录变化

每个实验运行后，都会在各自的 `results/` 子目录中生成输出。以实验 1 为例：

**运行前**：
```
experiment_01_toy_diffusion/
├── README.md
└── run.py
```

**运行后**（`python run.py` 执行完毕）：
```
experiment_01_toy_diffusion/
├── README.md
├── run.py
└── results/                          # [自动创建]
    ├── forward_two_moons.png
    ├── forward_swiss_roll.png
    ├── forward_gaussian_mixture.png
    ├── loss_two_moons.png
    ├── loss_swiss_roll.png
    ├── loss_gaussian_mixture.png
    ├── gen_two_moons.png
    ├── gen_swiss_roll.png
    ├── gen_gaussian_mixture.png
    └── results.json                  # 所有数值指标汇总
```

### 2.4 各实验结果文件一览

| 实验 | results/ 目录下生成的文件 | 说明 |
|------|---------------------------|------|
| **实验 1** | `forward_*.png`, `gen_*.png`, `loss_*.png`, `results.json` | 前向过程、生成对比、损失曲线、指标 |
| **实验 2** | `Q_heatmap_*.png`, `loss_*.png`, `results.json` | Q矩阵热力图、损失曲线、三种corruption指标 |
| **实验 3** | `mask_ratio_check.png`, `tau_ablation.png`, `recovery_by_time.png`, `loss.png`, `results.json` | mask ratio验证、tau消融、时间恢复、损失 |
| **实验 4** | `train_loss.png`, `val_ppl.png`, `maskstep_comparison.png`, `noise_schedule_comparison.png` | 训练/验证曲线、mask-step对比、噪声日程对比 |
| **实验 5** | `steps_ablation.png`, `token_selection_*.png`, `unmask_*.png` | 步数消融、token选择策略、unmask策略 |
| **实验 6** | `baseline_train_loss.png`, `baseline_val_ppl.png` | 三模型训练损失与验证PPL对比 |
| **实验 7** | `accuracy_comparison.png`, `edr_comparison.png`, `results.json` | 准确率对比、EDR对比、完整数值 |
| **实验 8** | `token_type_comparison.png`, `results.json` | 三种token类型准确率对比 |
| **实验 9** | `topk_accuracy.png`, `reliability_diagram.png`, `demo_repair.txt`, `results.json` | Top-K准确率、可靠性图、修复样例 |
| **实验 10** | `comparison.png`, `results.json` | D3PM vs CTMC 多维度对比 |

### 2.5 实验 4-6 的特殊路径

实验 4-6 在 MDLM 官方代码仓库中运行，其 checkpoint 和输出默认存储在官方仓库目录下：

```
/path/to/mdlm/
├── configs/                          # 需从本项目复制配置进来
│   ├── data/tinystories.yaml
│   └── model/small_tinystories.yaml
├── outputs/                          # [训练后] checkpoint 和日志
│   └── <timestamp>_mdlm-tinystories/
│       ├── checkpoints/              # 模型权重 (.ckpt)
│       ├── logs/                     # 训练日志
│       └── wandb/                    # 可视化记录
└── checkpoints/                      # [手动下载] 预训练权重
    ├── mdlm_step100.ckpt
    └── sedd.ckpt
```

**关键路径说明**：

| 路径 | 何时存在 | 用途 |
|------|----------|------|
| `mdlm/outputs/` | 训练完成后 | 存放训练产生的 checkpoint 和日志 |
| `mdlm/checkpoints/` | 手动创建/下载 | 存放预训练或自己复制的 checkpoint |
| 本项目的 `experiment_04/configs/` | 始终存在 | 配置文件模板，需复制到官方仓库 |
| 本项目的 `experiment_04/results/` | 手动整理后 | 建议将官方仓库输出整理后复制到此 |

**建议**：实验 4-6 运行后，将关键图表从 `mdlm/outputs/` 复制回本项目的对应 `experiment_0*/results/` 目录，便于统一管理。

---

## 3. 数据获取

### 3.1 自动下载 TinyStories

```bash
cd reproduce_mdlm_experiments/data
python prepare_data.py
```

**运行前**：
```
data/
└── prepare_data.py
```

**运行后**：
```
data/
├── prepare_data.py
├── cache/                            # [新增] HuggingFace 数据集缓存 (~500MB)
├── val_texts_100.pt                  # [新增] 100条验证文本
└── val_set_100.pt                    # [新增] 100条验证 token IDs
```

首次运行会自动从 HuggingFace 下载 `roneneldan/TinyStories`，下载后：
- 训练集：2,119,719 条
- 验证集：21,990 条
- 固定 100 条验证样本：`data/val_texts_100.pt`

### 3.2 下载预训练权重（跳过训练时使用）

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

**运行后新增**：
```
checkpoints/                          # [新增] 手动创建或自动创建
├── mdlm_step100.ckpt                # ~500MB
└── sedd.ckpt                        # ~500MB
```

### 3.3 数据存储位置详解

运行 `python data/prepare_data.py` 后，所有数据都存储在 `data/` 目录下，分为三个部分：

| 文件/目录 | 内容 | 大小 | 下游使用方 |
|-----------|------|------|-----------|
| `data/cache/` | HuggingFace 原始数据集缓存（含 train/validation 全部数据） | ~500MB | 实验 4-6（MDLM 训练时直接读取） |
| `data/val_texts_100.pt` | 100 条验证文本（`list[str]`，原始字符串） | ~50KB | 实验 7（mask 修复输入）、实验 8（token 类型分组）、实验 9（Top-K 修复） |
| `data/val_set_100.pt` | 100 条验证文本的 GPT-2 token IDs（`list[list[int]]`） | ~20KB | 实验 7-9 中需要预先编码的场景 |

**关键说明**：

- `cache/` 中存放的是 HuggingFace `datasets` 库的标准缓存格式，实验 4-6 在 MDLM 官方代码中训练时会通过 `load_dataset("roneneldan/TinyStories", cache_dir=...)` 自动读取。
- `val_texts_100.pt` 和 `val_set_100.pt` 是下游评估实验（7-9）的固定输入。它们通过 `seed=1` 从验证集中随机抽取，确保复现结果与论文一致。
- 只要 `val_texts_100.pt` 和 `val_set_100.pt` 存在，实验 7-9 即可正常运行，无需重复下载完整数据集。

### 3.4 使用本地数据

如果你已经下载过 TinyStories 或者有离线数据集，以下四种方式可以避免重复下载：

#### 方式一：利用 HuggingFace 缓存（推荐）

`load_dataset` 会自动检测 `cache_dir` 中是否已有数据。将已有的 TinyStories 缓存复制到项目目录：

```bash
# 将已有缓存复制到项目 data/cache/ 目录
cp -r /path/to/your/tinystories_cache/*  reproduce_mdlm_experiments/data/cache/
```

然后正常运行：
```bash
cd reproduce_mdlm_experiments/data
python prepare_data.py
```

脚本会检测到缓存存在，跳过下载，仅执行验证集抽取和 tokenizer 编码。

#### 方式二：修改脚本指向本地路径

如果你有离线数据集文件（如 parquet/json 格式），修改 `data/prepare_data.py`：

```python
# 原代码（从 HuggingFace 下载）
# dataset = load_dataset("roneneldan/TinyStories", cache_dir=CACHE_DIR)

# 改为加载本地文件
dataset = load_dataset("parquet", data_files={
    "train": "/path/to/your/train.parquet",
    "validation": "/path/to/your/validation.parquet"
})
```

支持的格式包括 `parquet`、`json`、`csv` 等，具体取决于你的本地数据格式。

#### 方式三：仅生成本地验证集（跳过完整下载）

如果你只需要那 100 条验证样本用于下游实验（7-9），可以直接用自己的文本生成：

```python
import torch
from transformers import GPT2Tokenizer

# 用自己的文本替换（至少 100 条）
my_texts = ["你的文本1...", "你的文本2...", ...]

# 保存为相同格式
torch.save(my_texts[:100], "data/val_texts_100.pt")

# 同时生成 tokenized 版本
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
encoded = [tokenizer.encode(t, max_length=256, truncation=True) for t in my_texts[:100]]
torch.save(encoded, "data/val_set_100.pt")
```

生成后实验 7-9 即可直接读取，无需运行 `prepare_data.py`。

#### 方式四：设置 HuggingFace 镜像（国内网络）

如果因网络问题无法连接到 HuggingFace 官方站点，使用镜像站：

```bash
export HF_ENDPOINT=https://hf-mirror.com
cd data && python prepare_data.py
```

设置后 `datasets` 库会自动从镜像站下载数据，无需修改代码。

### 3.5 各实验数据依赖关系

| 实验编号 | 实验名称 | 数据依赖 | 数据来源 |
|----------|----------|----------|----------|
| 1 | 连续二维 Diffusion | 无（代码内生成 synthetic 数据） | 运行时生成 |
| 2 | D3PM 离散 Toy | 无（代码内生成 synthetic 数据） | 运行时生成 |
| 3 | CTMC 与 Tau-Leaping | 无（代码内生成 synthetic 数据） | 运行时生成 |
| 4 | MDLM 模型训练 | TinyStories 完整数据集 | `data/cache/` 或 HuggingFace |
| 5 | 采样策略消融 | 实验 4 产出的 checkpoint | `checkpoints/mdlm_step100.ckpt` |
| 6 | 基线对比训练 | TinyStories 完整数据集 | `data/cache/` 或 HuggingFace |
| 7 | Mask 修复 | 100 条验证文本 | `data/val_texts_100.pt` |
| 8 | Token 类型分组 | 100 条验证文本 | `data/val_texts_100.pt` |
| 9 | Top-K 交互式修复 | 100 条验证文本 | `data/val_texts_100.pt` |
| 10 | D3PM vs CTMC | 无（代码内生成 synthetic 数据） | 运行时生成 |

**总结**：
- **实验 1-3、10**：完全自包含，不依赖任何外部数据，直接运行即可。
- **实验 4、6**：需要 TinyStories 完整数据集（通过 `data/cache/` 或 HuggingFace 获取）。
- **实验 7-9**：只需要 100 条验证样本（`data/val_texts_100.pt`），可通过上述四种方式获取。
- **实验 5**：依赖实验 4 产出的 checkpoint，不直接依赖数据集。

---

## 4. Phase 1：Toy 实验快速启动

Toy 实验（1, 2, 3, 10）**完全独立**，不依赖 MDLM 官方代码，安装好 `requirements.txt` 后即可运行。

### 4.1 实验 1：连续二维 Diffusion

```bash
cd experiment_01_toy_diffusion
python run.py
```

**运行前**：
```
experiment_01_toy_diffusion/
├── README.md
└── run.py
```

**运行后**（约 10-15 分钟）：
```
experiment_01_toy_diffusion/
├── README.md
├── run.py
└── results/                          # [自动创建]
    ├── forward_two_moons.png         # 前向加噪可视化
    ├── forward_swiss_roll.png
    ├── forward_gaussian_mixture.png
    ├── loss_two_moons.png            # 训练损失曲线
    ├── loss_swiss_roll.png
    ├── loss_gaussian_mixture.png
    ├── gen_two_moons.png             # 真实 vs 生成样本对比
    ├── gen_swiss_roll.png
    ├── gen_gaussian_mixture.png
    └── results.json                  # MMD、Sliced Wasserstein 等指标
```

**关键观察**：MMD 在 1e-3 量级说明恢复质量较高

### 4.2 实验 2：D3PM 离散 Toy

```bash
cd experiment_02_d3pm_toy
python run.py
```

**运行前**：
```
experiment_02_d3pm_toy/
├── README.md
└── run.py
```

**运行后**：
```
experiment_02_d3pm_toy/
├── README.md
├── run.py
└── results/                          # [自动创建]
    ├── Q_heatmap_uniform.png         # 三种 Q 矩阵热力图
    ├── Q_heatmap_absorbing.png
    ├── Q_heatmap_structured.png
    ├── loss_uniform.png              # 训练损失曲线
    ├── loss_absorbing.png
    ├── loss_structured.png
    └── results.json                  # 三种 corruption 的准确率对比
```

**关键观察**：structured corruption 的 token recovery 最高（~85%），uniform 最低（~70%）

### 4.3 实验 3：CTMC 与 Tau-Leaping

```bash
cd experiment_03_ctmc_toy
python run.py
```

**运行前**：
```
experiment_03_ctmc_toy/
├── README.md
└── run.py
```

**运行后**：
```
experiment_03_ctmc_toy/
├── README.md
├── run.py
└── results/                          # [自动创建]
    ├── mask_ratio_check.png          # 闭式 mask ratio 验证
    ├── tau_ablation.png             # tau 与误差/时间权衡
    ├── recovery_by_time.png         # 恢复难度随时间变化
    ├── loss.png                      # 训练损失曲线
    └── results.json                  # 数值指标与 tau 消融数据
```

**关键观察**：tau=0.01 时 TV distance 约 0.022，是精度和效率的较好平衡

### 4.4 实验 10：D3PM vs CTMC 对比

```bash
cd experiment_10_d3pm_vs_ctmc
python run.py
```

**运行前**：
```
experiment_10_d3pm_vs_ctmc/
├── README.md
└── run.py
```

**运行后**（约 20-30 分钟）：
```
experiment_10_d3pm_vs_ctmc/
├── README.md
├── run.py
└── results/                          # [自动创建]
    ├── comparison.png                # 多维度对比图（6个子图）
    └── results.json                  # 三个序列长度×两种方法的完整指标
```

---

## 5. Phase 2：MDLM 模型训练

### 5.1 复制配置文件

```bash
cd /path/to/mdlm

# 复制 TinyStories 配置
cp /path/to/reproduce_mdlm_experiments/experiment_04/configs/tinystories.yaml \
   configs/data/

cp /path/to/reproduce_mdlm_experiments/experiment_04/configs/small_tinystories.yaml \
   configs/model/
```

### 5.2 训练 MDLM（实验 4）

```bash
cd /path/to/mdlm
bash /path/to/reproduce_mdlm_experiments/experiment_04/scripts/train_mdlm.sh
```

**运行前**：官方仓库无 TinyStories 相关配置
**运行后**：
```
/path/to/mdlm/
├── configs/
│   ├── data/tinystories.yaml         # [复制进来]
│   └── model/small_tinystories.yaml  # [复制进来]
└── outputs/                          # [训练后自动生成]
    └── <timestamp>_mdlm-tinystories-baseline/
        ├── checkpoints/
        │   └── last.ckpt            # 训练好的模型权重
        └── logs/
```

**预期行为**：
- 训练 loss 整体下降
- 验证 PPL 早期可能有尖峰，随后回落
- 最终 PPL 约在 30-40 之间

**运行时间**：单卡 A100 约 2-3 小时

**建议**：训练完成后，将 `outputs/` 中的关键图表复制到本项目的 `experiment_04_mdlm_training/results/` 目录统一管理。

### 5.3 训练基线对比（实验 6）

```bash
cd /path/to/mdlm
bash /path/to/reproduce_mdlm_experiments/experiment_06/scripts/train_all_baselines.sh
```

这会依次训练 AR、SEDD、MDLM 三个模型。

**运行后**：
```
/path/to/mdlm/outputs/                # 三个子目录，分别对应三个模型
├── <timestamp>_ar-tinystories/
│   └── checkpoints/last.ckpt
├── <timestamp>_sedd-tinystories/
│   └── checkpoints/last.ckpt
└── <timestamp>_mdlm-tinystories-ref/
    └── checkpoints/last.ckpt
```

**运行时间**：单卡 A100 约 6-8 小时（三个模型）

---

## 6. Phase 3：采样策略消融

### 6.1 准备

确保已有 `checkpoints/mdlm_step100.ckpt`（实验 4 训练产出或预训练权重）。

### 6.2 运行采样步数消融（实验 5.1）

```bash
cd /path/to/mdlm
bash /path/to/reproduce_mdlm_experiments/experiment_05/scripts/ablation_steps.sh
```

**运行后输出**：官方仓库 `outputs/` 下生成多个子目录，每个对应一个采样步数配置。

### 6.3 运行 Token Selection 消融（实验 5.2）

```bash
cd /path/to/mdlm
bash /path/to/reproduce_mdlm_experiments/experiment_05/scripts/ablation_token_selection.sh
```

**注意**：需要在 `diffusion.py` 的采样循环中集成 `custom_sampler.py` 的策略函数。

### 6.4 运行 Unmask 策略消融（实验 5.3）

```bash
cd /path/to/mdlm
bash /path/to/reproduce_mdlm_experiments/experiment_05/scripts/ablation_unmask.sh
```

---

## 7. Phase 4：下游修复评估

### 7.1 实验 7：Mask 修复

```bash
cd reproduce_mdlm_experiments/experiment_07_mask_repair
python run.py
```

**运行前**：
```
experiment_07_mask_repair/
├── README.md
└── run.py
```

**运行后**：
```
experiment_07_mask_repair/
├── README.md
├── run.py
└── results/                          # [自动创建]
    ├── accuracy_comparison.png       # 三模型 × 九任务准确率柱状图
    └── results.json                  # 完整数值结果（三模型 × 九任务 × 多指标）
```

**注意**：当前 `run.py` 使用占位修复函数。实际使用时需要加载真实 checkpoint 并替换 `dummy_repair` 函数。

### 7.2 实验 8：Token 类型分组

```bash
cd reproduce_mdlm_experiments/experiment_08_token_type
python run.py
```

**运行前**：
```
experiment_08_token_type/
├── README.md
└── run.py
```

**运行后**：
```
experiment_08_token_type/
├── README.md
├── run.py
└── results/                          # [自动创建]
    ├── token_type_comparison.png     # English/Punctuation/Rare 三类对比
    └── results.json                  # 分组准确率数值
```

### 7.3 实验 9：Top-K 交互式修复

```bash
cd reproduce_mdlm_experiments/experiment_09_topk_repair
python run.py
```

**运行前**：
```
experiment_09_topk_repair/
├── README.md
└── run.py
```

**运行后**：
```
experiment_09_topk_repair/
├── README.md
├── run.py
└── results/                          # [自动创建]
    ├── topk_accuracy.png             # Top-1/3/5 准确率柱状图
    ├── reliability_diagram.png       # 置信度-准确率可靠性图
    ├── demo_repair.txt               # 人工辅助修复文本样例
    └── results.json                  # Top-K 准确率和可靠性数据
```

---

## 8. 常见问题排查

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

### 问题 3：找不到结果目录

**现象**：运行后不知道结果存在哪里

**解决**：所有实验的结果都在各自 `experiment_*/results/` 目录下。如果未找到，检查：
1. 脚本是否成功执行完毕（未报错中断）
2. 当前工作目录是否正确（应在实验目录或项目根目录运行）
3. 是否有写权限（`results/` 目录会在首次运行时自动创建）

### 问题 4：缺少 tokenizer

**现象**：`OSError: Can't load tokenizer`

**解决**：
```python
from transformers import GPT2Tokenizer
# 首次使用会自动下载
```

### 问题 5：实验 4-6 找不到配置

**现象**：`Could not load 'configs/data/tinystories.yaml'`

**解决**：确保已将配置文件从本项目的 `experiment_04/configs/` 复制到官方 mdlm 仓库的 `configs/` 目录

---

## 9. 结果分析指南

### 9.1 如何判断实验是否成功？

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

### 9.2 关键指标速查

| 指标 | 含义 | 越高越好？ |
|------|------|-----------|
| Mask Accuracy | mask 位置正确恢复率 | 是 |
| EDR | 编辑距离缩减率 | 是 |
| Gen PPL | 生成文本困惑度 | 否（越低越好）|
| TV Distance | 真实/生成分布差异 | 否 |
| Near Rate | 模板匹配率 | 是 |

---

*本手册与 reproduce_mdlm_experiments 仓库同步更新。如有问题请提交 Issue。*
