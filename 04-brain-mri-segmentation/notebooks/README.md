
---

## 📊 实验设计总览

所有实验保持以下固定参数：
- **网络**：UNet（`init_features=32`）
- **数据集**：同一切片划分（随机种子固定）
- **评价指标**：验证集平均 Dice 相似系数（DSC）
- **训练 epochs**：100（early stopping 未开启）
- **Batch size**：16
- **初始学习率**：1e-4（Adam/AdamW）

每轮实验记录训练/验证损失、每病人 DSC，并保存最佳模型权重。

---

## 📓 Notebook 详述

### 1. `mri-skimage-基线代码.ipynb`
- **说明**：直接运行来自 GitHub 的基线 UNet 代码（固定结构：MaxPool + ConvTranspose，损失：Dice，优化器：Adam）。
- **目的**：建立性能基准（Baseline DSC ≈ 0.9087）。

### 2. `mri-skimage-compare.ipynb`
- **实验组**（单变量对比，保持其他为基线）：
  - **下采样**：`Strided Conv`（替换 MaxPool）
  - **优化器**：`AdamW`
  - **混合精度训练**：`AMP`
  - **数据增强**：`弹性形变（Elastic）`
- **目的**：评估各单一改进对分割性能的影响。

### 3. `mri-skimage-compare2.ipynb`
- **实验组**（修复了前一轮的 bug）：
  - **损失函数**：`Focal Loss`（γ=2.0）
  - **上采样**：`Bilinear`（替换 ConvTranspose）
  - **组合实验**：`弹性形变 + Focal Loss + AdamW + AMP`
- **目的**：继续探索损失函数和上采样方式的影响，并测试组合效果。

### 4. `mri-skimage-compare3.ipynb`
- **实验组**：
  - **上采样**：`PixelShuffle`（替换 ConvTranspose）
- **目的**：测试另一种上采样方式的性能。

### 5. `mri-skimage-compare4.ipynb`
- **实验组**：
  - **组合**：`PixelShuffle + AdamW`
- **目的**：验证“最佳上采样”和“最佳优化器”结合是否能产生叠加效果。

### 6. `mri-skimage-compare5-DiceBCE.ipynb`
- **实验组**（基于当前最优配置 `PixelShuffle + Adam`）：
  - **损失函数**：`Dice Loss + BCE Loss` 组合
    - 默认权重（Dice:1.0, BCE:0.5）
    - 等权重（1.0, 1.0）
    - 偏重 Dice（1.0, 0.2）
    - 偏重 BCE（0.4, 0.6）
- **目的**：探索组合损失能否进一步提升 DSC，并找到最佳权重配比。

---

## 🔧 使用说明

所有 Notebook 均通过调用 `train_exp.py` 进行训练，可通过命令行参数灵活配置。例如：

```bash
python train_exp.py --loss dice_bce --dice_weight 1.0 --bce_weight 0.5 --ckpt_name DiceBCE_default
```

