# Diffusion & MDLM 实验复现项目

> 本项目按照《Diffusion 与 MDLM 实验流程和结果分析》文档（DDPM3），为每个实验提供独立的子文件夹，包含完整的代码实现、运行说明和结果分析。
>
> **参考仓库**：
> - 官方 MDLM：`https://github.com/kuleshov-group/mdlm`
> - 官方离散扩散 Guidance：`https://github.com/kuleshov-group/discrete-diffusion-guidance`
> - 参考复现（含预训练权重）：`https://github.com/goosemaths/mdlm`

---

## 项目结构

```
reproduce_mdlm_experiments/
├── README.md                          # 本文件
├── PRACTICE_GUIDE.md                  # 实践指导手册（手把手教程）
├── requirements.txt                   # Python 依赖
├── data/
│   └── prepare_data.py               # TinyStories 数据下载与验证集准备
│
├── experiment_01_toy_diffusion/       # 实验 1：连续二维 toy diffusion
├── experiment_02_d3pm_toy/            # 实验 2：D3PM 离散状态 toy
├── experiment_03_ctmc_toy/            # 实验 3：CTMC discrete diffusion
├── experiment_04_mdlm_training/       # 实验 4：MDLM 基础复现与训练稳定性
├── experiment_05_sampling_ablation/   # 实验 5：采样步数与固定步数策略消融
├── experiment_06_baseline_comparison/ # 实验 6：AR/SEDD/MDLM 基线对比
├── experiment_07_mask_repair/         # 实验 7：三模型下游 mask 修复稳定性
├── experiment_08_token_type/          # 实验 8：token 类型分组准确率
├── experiment_09_topk_repair/         # 实验 9：Top-K 交互式修复
└── experiment_10_d3pm_vs_ctmc/        # 实验 10：D3PM 与 CTMC 对比
```

每个实验文件夹包含：
- `README.md` — 实验原理、运行步骤、预期结果
- `run.py` 或 `scripts/` — 主实验代码（实验 1-3, 10 可独立运行）
- `results/` — 实验结果输出目录（运行后自动生成）

---

## 环境准备

### 1. 安装基础依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

```bash
cd data
python prepare_data.py
```

这会下载 TinyStories 数据集并准备 100 条固定验证样本。

### 3. 获取 MDLM 官方代码（实验 4-9 需要）

```bash
# 在项目根目录同级克隆官方仓库
cd ..
git clone https://github.com/kuleshov-group/mdlm.git
cd mdlm
conda env create -f requirements.yaml
conda activate mdlm
```

实验 4-9 的脚本会在官方代码基础上运行，具体配置见各实验 README。

### 4. 下载预训练权重（可选，跳过训练）

如果不想从头训练实验 4-6 的模型，可以直接下载 goosemaths 提供的 checkpoint：

```python
from huggingface_hub import hf_hub_download

repo_id = "goosemaths/tinystories_masked_diffusion_model_ckpt"
hf_hub_download(repo_id=repo_id, filename="mdlm_step100.ckpt", local_dir="checkpoints")
hf_hub_download(repo_id=repo_id, filename="sedd.ckpt", local_dir="checkpoints")
```

---

## 实验运行顺序

建议按以下顺序运行，后续实验可能依赖前面实验的 checkpoint：

| 阶段 | 实验 | 说明 | 独立运行？ |
|------|------|------|-----------|
| Phase 1 | 1, 2, 3, 10 | Toy 实验，验证基础扩散机制 | 完全独立 |
| Phase 2 | 4, 6 | 模型训练（或用预训练权重） | 依赖 MDLM 官方代码 |
| Phase 3 | 5 | 采样消融（需实验 4 的 checkpoint） | 依赖实验 4 |
| Phase 4 | 7, 8, 9 | 下游修复评估（需实验 4/6 的 checkpoint） | 依赖实验 4/6 |

**快速路径**：直接用预训练权重跳过 Phase 2，从 Phase 3 开始。

---

## 统一实验配置

| 配置项 | 值 |
|--------|-----|
| 数据集 | TinyStories |
| 训练样本数 | 2,119,719 |
| 验证样本数 | 21,990 |
| 下游评测样本数 | 100 条验证文本 |
| 随机种子 | 1 |
| 最大训练步数 | 5,000 |
| Batch 设置 | global=128, train=16, eval=16 |
| 上下文长度 | 256 |
| 模型 | AR / SEDD / MDLM |
| 采样步数消融 | 10, 20, 50, 100, 200, 500, 1000 |
| 修复任务 | Random Mask, OCR-like, Span Mask |

---

## 快速启动

```bash
# Phase 1: 运行所有 Toy 实验（约 10-30 分钟）
cd experiment_01_toy_diffusion && python run.py && cd ..
cd experiment_02_d3pm_toy && python run.py && cd ..
cd experiment_03_ctmc_toy && python run.py && cd ..
cd experiment_10_d3pm_vs_ctmc && python run.py && cd ..

# Phase 2: 训练 MDLM 和基线（约 2-4 小时，GPU 推荐）
cd ../mdlm  # 官方代码目录
cp reproduce_mdlm_experiments/experiment_04/configs/* configs/
bash reproduce_mdlm_experiments/experiment_04/scripts/train_mdlm.sh
bash reproduce_mdlm_experiments/experiment_06/scripts/train_all_baselines.sh

# Phase 3: 采样消融（约 1-2 小时）
bash reproduce_mdlm_experiments/experiment_05/scripts/ablation_steps.sh

# Phase 4: 下游评估
cd reproduce_mdlm_experiments
cd experiment_07_mask_repair && python run.py && cd ..
cd experiment_08_token_type && python run.py && cd ..
cd experiment_09_topk_repair && python run.py && cd ..
```

---

## 实践指导手册

详细的实践操作指南请阅读 [PRACTICE_GUIDE.md](PRACTICE_GUIDE.md)，包含：
- 每一步的详细操作说明
- 常见问题排查
- 结果解读方法
- 实验间的依赖关系

---

## 参考仓库

- **官方 MDLM**: https://github.com/kuleshov-group/mdlm
- **官方离散扩散 Guidance**: https://github.com/kuleshov-group/discrete-diffusion-guidance
- **参考复现（含预训练权重）**: https://github.com/goosemaths/mdlm
