# 图像自然语言描述生成实验

本实验基于 Flickr8k 数据集，实现并对比多种 Encoder-Decoder 模型与解码策略，完成图像自动描述生成任务。

---

## 📌 项目简介

- **任务**：输入一张图片，模型生成一句英文描述。
- **框架**：ResNet-50 编码器 + LSTM 解码器（含/无注意力机制）。
- **评估指标**：BLEU-4。
- **主要贡献**：
  - 实现 Baseline、Stronger Regularization、Attention、Scheduled Sampling、Encoder Fine-tuning 五种配置。
  - 对比 beam size（1/3/5/7）对 BLEU-4 的影响。
  - 最优模型（Attention + Fine-tuned Encoder + beam=3）在测试集上达到 **BLEU-4 = 0.2494**。

---

## 📂 数据集

- **Flickr8k**：8,000 张图片，每张 5 条人工标注描述。
- **划分**：Karpathy split（训练 6,000 / 验证 1,000 / 测试 1,000 张图片）。
- **预处理**：构建词表（含 `<pad>`, `<start>`, `<end>`, `<unk>`），编码 caption，生成 `vocab.json` 和 `captions_encoded.json`。

---

## 🧪 实验方法概览

| 模型 | 描述 | Attention | Scheduled Sampling | Fine-tune Encoder |
|------|------|-----------|---------------------|-------------------|
| Baseline | Show and Tell | ❌ | ❌ | ❌ |
| SR | Baseline + 更强正则化（LR=3e-4, dropout=0.3） | ❌ | ❌ | ❌ |
| Att | Show, Attend and Tell | ✅ | ❌ | ❌ |
| ASS | Attention + 延迟 Scheduled Sampling | ✅ | ✅（延迟开启） | ❌ |
| Finetune | Attention + Encoder Fine-tuning | ✅ | ❌ | ✅（LR=1e-5） |

- **训练细节**：batch size=32，Adam 优化器，CrossEntropyLoss（忽略 padding），基于验证 BLEU-4 保存最佳模型。
- **解码对比**：beam size = 1, 3, 5, 7。

---

## 📊 主要结果

### 测试集最佳 BLEU-4（各模型最优 beam）

| 模型 | 最佳 Beam Size | BLEU-4 |
|------|---------------|--------|
| Baseline | 3 | 0.2234 |
| SR | 3 | 0.2229 |
| Att | 7 | 0.2181 |
| ASS | 3 | 0.2314 |
| **Finetune** | **3** | **0.2494** |

- **Beam Search 有效**：多数模型 beam=3 时 BLEU 最高，过大（≥5）反而下降。
- **Attention 非稳定提升**：单独加 Attention 在测试集 greedy 结果低于 Baseline。
- **Scheduled Sampling** 和 **Encoder Fine-tuning** 贡献显著。

---

## 🗂️ 文件结构说明

```plaintext
实验6-图像自然语言描述生成/
├── data/
│   └── flickr8k-archive/          # 原始数据集
│       ├── Images/                # 所有图片
│       ├── captions.txt           # 原始标注
│       ├── vocab.json             # 预处理词表
│       └── captions_encoded.json  # 编码后的标注
├── output/                        # 训练好的模型检查点及评估结果
│   ├── beam_search_comparison_val.csv
│   ├── beam_search_comparison_test.csv
│   ├── Baseline/
│   │   └── Baseline_best_model.pt
│   ├── StrongerRegularization/
│   │   └── SR_best_model.pt
│   ├── Attention/
│   │   └── Att_best_model.pt
│   ├── Attention_ScheduledSampling/
│   │   └── ASS_best_model.pt
│   └── Finetune/
│       └── finetune_best_model.pt
├── models/                        # 模型定义（encoder, decoder, attention）
│   ├── encoder.py
│   ├── decoder.py
│   ├── attention.py
│   └── __init__.py
├── caption_dataset.py             # 数据集类与 collate_fn
├── utils.py                       # 工具函数（AverageMeter 等）
├── train.py                       # 基础训练脚本
├── train_exp.py                   # 实验性训练脚本（用于不同配置）
├── eval_checkpoints.py            # 模型评估脚本（支持 beam search）
├── captioning.ipynb               # 可视化与结果展示
└── flickr8k-*.ipynb               # 预处理、调试、实验记录等笔记本
```