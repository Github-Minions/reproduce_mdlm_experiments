# 实验 9：MDLM step=100 下的 Top-K 交互式修复

## 实验目的

将单一自动修复扩展为候选式修复：高置信位置直接自动填充，低置信位置展示 Top-K 候选供人工选择。验证 Top-1/3/5 Accuracy 和 reliability diagram。

## 实验设置

| 配置项 | 值 |
|--------|-----|
| 模型 | MDLM step=100 checkpoint |
| 置信度阈值 | 0.8（>=0.8 自动填充，<0.8 展示候选）|
| 候选数量 K | 1, 3, 5 |

## 运行

```bash
cd experiment_09_topk_repair
python run.py --checkpoint ../checkpoints/mdlm_step100.ckpt
```

## 预期结果

### Top-K 准确率

| 候选数 | Accuracy |
|--------|----------|
| Top-1 | 80.51% |
| **Top-3** | **92.01%** |
| Top-5 | 94.55% |

> Top-1 到 Top-3 提升 11.50 个百分点，Top-3 到 Top-5 仅提升 2.54 个百分点。

### Reliability Diagram

高置信区间的实际准确率更高，支持基于置信度的自动填充与人工候选分流。

### 人工辅助修复样例

```
Input:  Spot saw the shiny car and said, "Wow, Kitty, your car is so [MASK] and clean!"

High confidence auto-fill:
  car -> 0.93
  They -> 0.94
  friends -> 1.00

Low confidence candidates:
  clean / nice / shiny (0.23)
  polished / polish / value (0.61)

Decision rule: confidence >= 0.8 -> auto-fill; otherwise show Top-3 candidates
```

## 输出文件

- `results/topk_accuracy.png` — Top-K 准确率柱状图
- `results/reliability_diagram.png` — 置信度-准确率曲线
- `results/demo_repair.txt` — 人工辅助修复样例
