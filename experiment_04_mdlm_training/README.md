# 实验 4：MDLM 基础复现与训练稳定性

## 实验目的

验证 MDLM 的 mask 训练链路，观察训练曲线、验证困惑度，并比较不同训练 mask-step 和噪声日程的影响。

## 前置要求

本实验依赖 `kuleshov-group/mdlm` 官方代码库。请先克隆并配置：

```bash
# 在项目根目录的同级位置
cd /path/to/reproduce_mdlm_experiments/..
git clone https://github.com/kuleshov-group/mdlm.git
cd mdlm
conda env create -f requirements.yaml
conda activate mdlm
```

## 实验设置

| 配置项 | 值 |
|--------|-----|
| 数据集 | TinyStories |
| 模型 | DiT-small (768-dim, 12 layers, 12 heads) |
| 上下文长度 | 256 |
| 最大训练步数 | 5,000 |
| Batch | global=128, per-GPU=16, eval=16 |
| 参数化 | SUBS |
| 随机种子 | 1 |

## 运行步骤

### Step 1: 复制配置文件到官方仓库

```bash
# 从本实验目录复制配置到 mdlm 仓库
cp configs/tinystories.yaml /path/to/mdlm/configs/data/
cp configs/small_tinystories.yaml /path/to/mdlm/configs/model/
```

### Step 2: 训练 MDLM

```bash
cd /path/to/mdlm
bash /path/to/reproduce_mdlm_experiments/experiment_04/scripts/train_mdlm.sh
```

或手动运行：
```bash
python main.py \
  loader.global_batch_size=128 \
  loader.batch_size=16 \
  loader.eval_batch_size=16 \
  model=small_tinystories \
  data=tinystories \
  parameterization=subs \
  model.length=256 \
  seed=1 \
  trainer.max_steps=5000 \
  eval.compute_generative_perplexity=True \
  sampling.steps=100
```

### Step 3: 训练 mask-step 对比

```bash
bash scripts/train_maskstep_comparison.sh
```

### Step 4: 噪声日程对比

```bash
bash scripts/train_noise_schedule_comparison.sh
```

## 预期观察

- **训练 loss**：整体下降，后期进入缓慢收敛区间
- **验证 PPL**：早期存在尖峰，随后快速回落并保持稳定
- **不同 mask-step**：在当前小模型设置下，训练时 mask-step 对最终收敛趋势的影响相对有限
- **噪声日程**：linear noise 在收敛过程中表现出更大的局部波动

## 输出

- `results/train_loss.png` — 训练损失曲线
- `results/val_ppl.png` — 验证困惑度曲线
- `results/maskstep_comparison.png` — 不同 mask-step 的 loss 对比
- `results/noise_schedule_comparison.png` — 不同噪声日程的 loss 对比
