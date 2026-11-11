"""训练模型（支持命令行调参、早停、计划采样）。"""

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.optim as optim
import torchvision.transforms as transforms
from nltk.translate.bleu_score import corpus_bleu
from torch import Tensor, nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from caption_dataset import CaptionDataset, collate_fn
from models import Encoder, RNNDecoder, RNNDecoderWithAttention
from utils import AverageMeter

Decoder = RNNDecoder | RNNDecoderWithAttention


@dataclass
class Config:
    """Training config (可命令行覆盖)."""

    batch_size: int = 32
    """Batch size for training and testing."""
    num_workers: int = 4
    """Number of dataset loading workers."""
    learning_rate: float = 5e-4
    """Learning rate (of decoder)."""
    encoder_learning_rate: float = 1e-4
    """Encoder learning rate, used only when finetune_encoder=True."""
    weight_decay: float = 1e-4
    """Weight decay."""
    num_epochs: int = 20
    """Number of training epochs."""
    embed_dim: int = 512
    """Dimension of word embedding."""
    hidden_dim: int = 1024
    """Dimension of decoder hidden state."""
    grad_clip: float = 5.0
    """Grad clip norm."""
    dropout: float = 0.2
    """Dropout rate."""
    max_decode_len: int = 30
    """Maximum decode length."""
    beam_size: int = 1
    """Beam search size. 1 for greedy decoding."""
    device: str = "cuda"
    """Which device to use."""
    dataset_dir: str = "/kaggle/input/datasets/adityajn105/flickr8k"
    preprocess_dir: str = "/kaggle/input/datasets/fdiiiiii/flickr8k"
    """Preprocessed dataset directory."""
    output_dir: str = "output"
    """Output directory."""
    finetune_encoder: bool = False
    """Fine-tune encoder or not."""
    use_attention: bool = False
    """Use attention or not."""
    use_scheduled_sampling: int = 0
    """Whether to use scheduled sampling. 0 = pure teacher forcing, 1 = scheduled sampling."""
    teacher_forcing_ratio: float = 1.0
    """Current teacher forcing ratio."""
    early_stop_patience: int = 0
    """Early stopping patience (BLEU). 0 to disable."""
    # 不可通过命令行直接设置的内部字段
    vocab_size: int = field(default=0, init=False)


def parse_args(default_cfg: Config) -> Config:
    """解析命令行参数，覆盖默认配置。"""
    parser = argparse.ArgumentParser(
        description="Train image captioning model with various options."
    )
    parser.add_argument("--batch_size", type=int, default=default_cfg.batch_size)
    parser.add_argument("--num_workers", type=int, default=default_cfg.num_workers)
    parser.add_argument("--learning_rate", type=float, default=default_cfg.learning_rate)
    parser.add_argument("--encoder_learning_rate", type=float, default=default_cfg.encoder_learning_rate)
    parser.add_argument("--weight_decay", type=float, default=default_cfg.weight_decay)
    parser.add_argument("--num_epochs", type=int, default=default_cfg.num_epochs)
    parser.add_argument("--embed_dim", type=int, default=default_cfg.embed_dim)
    parser.add_argument("--hidden_dim", type=int, default=default_cfg.hidden_dim)
    parser.add_argument("--grad_clip", type=float, default=default_cfg.grad_clip)
    parser.add_argument("--dropout", type=float, default=default_cfg.dropout)
    parser.add_argument("--max_decode_len", type=int, default=default_cfg.max_decode_len)
    parser.add_argument("--beam_size", type=int, default=default_cfg.beam_size)
    parser.add_argument("--device", type=str, default=default_cfg.device)
    parser.add_argument("--dataset_dir", type=str, default=default_cfg.dataset_dir)
    parser.add_argument("--preprocess_dir", type=str, default=default_cfg.preprocess_dir)
    parser.add_argument("--output_dir", type=str, default=default_cfg.output_dir)
    parser.add_argument("--finetune_encoder", type=int, default=int(default_cfg.finetune_encoder))
    parser.add_argument("--use_attention", type=int, default=int(default_cfg.use_attention))
    parser.add_argument("--use_scheduled_sampling", type=int, default=default_cfg.use_scheduled_sampling)
    parser.add_argument("--teacher_forcing_ratio", type=float, default=default_cfg.teacher_forcing_ratio)
    parser.add_argument("--early_stop_patience", type=int, default=default_cfg.early_stop_patience)

    args = parser.parse_args()
    # 将解析结果更新到 Config 实例
    for key, value in vars(args).items():
        if hasattr(default_cfg, key):
            if key in ("finetune_encoder", "use_attention"):
                setattr(default_cfg, key, bool(value))
            else:
                setattr(default_cfg, key, value)
    return default_cfg


def load_models(vocab_size: int, cfg: Config):
    """加载 encoder 和 decoder。"""
    device = cfg.device
    encoder = Encoder(encoded_size=(7, 7), finetune=cfg.finetune_encoder).to(device)

    print("Probing feature dimension with dummy input...")
    dummy_features: Tensor = encoder(torch.zeros(1, 3, 224, 224, device=device))
    feature_dim = dummy_features.shape[-1]
    print(f"Feature dimension: {feature_dim}")

    if cfg.use_attention:
        decoder = RNNDecoderWithAttention(
            cfg.embed_dim, cfg.hidden_dim, vocab_size, feature_dim, dropout=cfg.dropout
        ).to(device)
    else:
        decoder = RNNDecoder(
            cfg.embed_dim, cfg.hidden_dim, vocab_size, feature_dim, dropout=cfg.dropout
        ).to(device)
    return encoder, decoder

def get_teacher_forcing_ratio(epoch: int, use_scheduled_sampling: int) -> float:
    """Return teacher forcing ratio for current epoch."""
    if not use_scheduled_sampling:
        return 1.0

    if epoch < 5:
        return 1.0
    elif epoch < 10:
        return 0.9
    else:
        return 0.8

def train_one_epoch(
    epoch: int,
    encoder: Encoder,
    decoder: Decoder,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizers: list[optim.Optimizer],
    schedulers: list[ReduceLROnPlateau],
    cfg: Config,
):
    """训练一个 epoch，支持 scheduled sampling。"""
    device = cfg.device
    if cfg.finetune_encoder:
        encoder.train()
    else:
        encoder.eval()
    decoder.train()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    use_attention = cfg.use_attention

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{cfg.num_epochs}")
    for imgs, captions, lengths in pbar:
        imgs: Tensor = imgs.to(device)
        captions: Tensor = captions.to(device)
        B, T = captions.shape

        # 提取图像特征
        features: Tensor = encoder(imgs)  # (B, num_pixels, feat_dim)

        # 初始化 LSTM 状态
        h, c = decoder.init_hidden_state(features)  # (B, hidden_dim)

        # 存储所有时间步的预测 logits
        outputs_list: list[Tensor] = []

        # 初始输入：<start> token
        input_tokens = captions[:, 0]  # (B,)
        prev_pred_tokens = None  # 上一步预测的词索引

        tf_ratio = get_teacher_forcing_ratio(epoch, cfg.use_scheduled_sampling)
        cfg.teacher_forcing_ratio = tf_ratio

        # 逐时间步计算
        for t in range(T - 1):  # 输出长度比输入少 1
            # Scheduled Sampling
            if t == 0:
                # 第一步强制使用 <start>
                current_tokens = captions[:, 0]
            else:
                # 真实词和预测词索引
                real_tokens = captions[:, t]

                if tf_ratio >= 1.0:
                    current_tokens = real_tokens
                else:
                    mask = torch.rand(B, device=device) < tf_ratio
                    pred_tokens = prev_pred_tokens.detach()
                    current_tokens = torch.where(mask, real_tokens, pred_tokens)
                
            # 词嵌入
            embed = decoder.embedding(current_tokens)  # (B, embed_dim)

            # Attention 分支
            if use_attention:
                context, _ = decoder.attention(h, features)
                lstm_input = torch.cat([embed, context], dim=1)
            else:
                lstm_input = embed

            # LSTM 步进
            h, c = decoder.lstm_cell(lstm_input, (h, c))
            h_dropped = decoder.dropout(h)
            logits = decoder.fc(h_dropped)  # (B, vocab_size)

            outputs_list.append(logits)
            prev_pred_tokens = logits.argmax(dim=1)  # 用于下一步采样

        # 堆叠输出 (B, T-1, vocab_size)
        outputs: Tensor = torch.stack(outputs_list, dim=1)
        # 目标：captions 去掉第一个 <start>
        targets = captions[:, 1:].contiguous().reshape(-1)
        outputs = outputs.reshape(-1, outputs.shape[2])  # (B*(T-1), vocab_size)

        loss: Tensor = criterion(outputs, targets)

        # 反向传播与优化
        for opt in optimizers:
            opt.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=cfg.grad_clip)
        if cfg.finetune_encoder:
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, encoder.parameters()),
                max_norm=cfg.grad_clip,
            )

        for opt in optimizers:
            opt.step()

        # 计算准确率（忽略 pad）
        with torch.no_grad():
            targets_not_padding = targets != 0
            num_targets = targets_not_padding.long().sum().item()
            num_correct = (
                torch.logical_and(outputs.argmax(-1) == targets, targets_not_padding)
                .long()
                .sum()
                .item()
            )

        loss_meter.update(loss.item(), num_targets)
        acc_meter.update(num_correct / num_targets, num_targets)
        pbar.set_postfix(
            {
                "Loss": f"{loss_meter.avg:.4f}",
                "Acc": f"{acc_meter.avg:.1%}",
                "LR": f"{optimizers[0].param_groups[0]['lr']:.2e}",
            }
        )


    return loss_meter.avg


def decode_caption(caption: list[int], rev_vocab: dict[int, str]) -> str:
    """将 token 序列转换为句子。"""
    words = []
    for m in caption:
        s = rev_vocab.get(m, "<unk>")
        if s in ("<start>", "<end>"):
            continue
        words.append(s)
    sentence = " ".join(words) + "."
    return sentence.capitalize()


@torch.no_grad()
def evaluate(
    encoder: Encoder,
    decoder: Decoder,
    dataloader: DataLoader,
    vocab: dict[str, int],
    cfg: Config,
):
    """验证/测试，计算 BLEU-4。"""
    device = cfg.device
    encoder.eval()
    decoder.eval()
    hypotheses: list[list[int]] = []
    references: list[list[list[int]]] = []

    for imgs, captions, lengths, all_captions in tqdm(dataloader, desc="Evaluating"):
        all_captions: list[list[list[int]]]
        imgs: Tensor = imgs.to(device)

        preds = decoder.generate_caption(
            encoder(imgs), vocab, max_len=cfg.max_decode_len, beam_size=cfg.beam_size
        )
        # 去掉 <start> 和 <end>
        for i, pred in enumerate(preds):
            preds[i] = pred[1:-1]
        for caps in all_captions:
            for j in range(len(caps)):
                caps[j] = caps[j][1:-1]
        hypotheses.extend(preds)
        references.extend(all_captions)

    print("Calculating BLEU score...")
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    return {"bleu-4": bleu4}


def main():
    # 使用默认配置，再通过命令行覆盖
    cfg = parse_args(Config())

    if not torch.cuda.is_available() and cfg.device == "cuda":
        cfg.device = "cpu"
        print("Warning: CUDA not available, using CPU.")

    print("Effective configuration:")
    print(json.dumps(asdict(cfg), indent=2, ensure_ascii=False))

    timestamp = datetime.now().strftime(r"%Y%m%d-%H%M%S")
    output_dir = Path(cfg.output_dir) / timestamp
    print(f"Output directory: {output_dir}")

    # 数据增强
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 数据集
    train_dataset = CaptionDataset(
        cfg.dataset_dir, preprocess_dir=cfg.preprocess_dir,
        split="train", transform=train_transform, cache=False
    )
    vocab = train_dataset.vocab
    vocab_size = len(vocab)
    val_dataset = CaptionDataset(
        cfg.dataset_dir, preprocess_dir=cfg.preprocess_dir,
        split="val", transform=val_transform, cache=False
    )
    print(
        f"Dataset: train={len(train_dataset):,} captions, val={len(val_dataset):,} captions"
    )

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=4, collate_fn=collate_fn, pin_memory=True,
    )

    # 模型
    encoder, decoder = load_models(vocab_size, cfg)

    # 损失与优化器
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # pad=0
    optimizers = [
        optim.Adam(decoder.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    ]
    # 额外给 encoder 加一个 optimizer
    if cfg.finetune_encoder:
        optimizers.append(
            optim.Adam(
                filter(lambda p: p.requires_grad, encoder.parameters()),
                lr=cfg.encoder_learning_rate,
                weight_decay=1e-5,
            )
        )
    # scheduler 根据 validation BLEU-4 调整，而不是 train loss
    schedulers = [
        ReduceLROnPlateau(
            opt, 
            mode="max", 
            factor=0.5, 
            patience=3
        ) 
        for opt in optimizers
    ]

    # 训练历史记录
    history = {
        "epoch": [],
        "train_loss": [],
        "val_bleu4": [],
        "learning_rate": [],
        "teacher_forcing_ratio": [],
    }
    # 确保输出目录存在（提前创建，避免第一次未保存模型时无法写入 history）
    output_dir.mkdir(parents=True, exist_ok=True)

    # 早停相关变量
    best_bleu4 = 0.0
    epochs_without_improvement = 0

    for epoch in range(cfg.num_epochs):
        train_loss = train_one_epoch(
            epoch, encoder, decoder, train_loader, criterion, optimizers, schedulers, cfg
        )
        print(f"Epoch {epoch+1:2d}/{cfg.num_epochs}  Train loss: {train_loss:.4f}")

        scores = evaluate(encoder, decoder, val_loader, vocab, cfg)
        bleu4 = scores["bleu-4"]
        print(f"BLEU-4: {bleu4:.4f}")

        for sched in schedulers:  # 当 BLEU-4 连续几轮不提升时，先降学习率，再考虑早停
            sched.step(bleu4)

        # 记录当前 epoch 指标
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_bleu4"].append(bleu4)
        history["learning_rate"].append(optimizers[0].param_groups[0]['lr'])
        history["teacher_forcing_ratio"].append(cfg.teacher_forcing_ratio)
        
        # 保存最佳模型
        if bleu4 > best_bleu4:
            best_bleu4 = bleu4
            epochs_without_improvement = 0
            # 目录已提前创建，但这里保留也无害
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / "config.json", "w") as f:
                json.dump(asdict(cfg), f, indent=2, ensure_ascii=True)
            torch.save(
                {
                    "config": asdict(cfg),
                    "encoder": encoder.state_dict(),
                    "decoder": decoder.state_dict(),
                },
                output_dir / "best_model.pt",
            )
            print(
                f"✓ Best model saved (epoch {epoch+1}, BLEU-4: {best_bleu4:.4f})"
            )
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epoch(s).")

        # 早停检查
        if cfg.early_stop_patience > 0 and epochs_without_improvement >= cfg.early_stop_patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # 保存训练历史到 JSON
    with open(output_dir / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {output_dir / 'train_history.json'}")

    print(f"Training finished. Best BLEU-4: {best_bleu4:.4f}")
    return best_bleu4


if __name__ == "__main__":
    main()