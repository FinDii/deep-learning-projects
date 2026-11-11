"""训练模型。"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
import torchvision.transforms as transforms
from nltk.translate.bleu_score import corpus_bleu
from torch import Tensor, nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from datasets import CaptionDataset, collate_fn
from models import Encoder, RNNDecoder, RNNDecoderWithAttention
from utils import AverageMeter

Decoder = RNNDecoder | RNNDecoderWithAttention


@dataclass
class Config:
    """Training config."""

    batch_size: int = 32
    """Batch size for training and testing."""
    num_workers: int = 12
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
    """Maximum decode length. Default: 30."""
    beam_size: int = 1
    """Beam search. 1 for greedy decoding. Default: 1."""
    device: str = "cuda"
    """Which device to use."""
    # TODO set directory to your own path
    dataset_dir: str = "data/flickr8k"
    """Dataset directory."""
    output_dir: str = "output"
    """Output directory."""
    finetune_encoder: bool = False
    """Fine-tune encoder or not."""
    use_attention: bool = False
    """Use attention or not."""


def load_models(vocab_size: int, cfg: Config):
    """Load encoder and decoder.

    Args:
        vocab_size (int): Vocabulary size.
        cfg (Config): config.

    Returns:
        tuple[Module, Module]: Encoder and decoder.
    """
    device = cfg.device
    encoder = Encoder(encoded_size=(7, 7), finetune=cfg.finetune_encoder).to(device)

    print("Trying to forward one dummy tensor to get the feature dimension...")
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
    """Train the model for one epoch.

    Args:
    """
    device = cfg.device
    if cfg.finetune_encoder:
        encoder.train()
    else:
        encoder.eval()
    decoder.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{cfg.num_epochs}")
    for batch_idx, (imgs, captions, lengths) in enumerate(pbar):
        imgs: Tensor
        captions: Tensor
        lengths: Tensor
        imgs = imgs.to(device)
        captions = captions.to(device)

        # 前向传播
        outputs: Tensor = decoder(encoder(imgs), captions)

        # 计算损失
        # 只计算非填充部分的损失
        targets = captions[:, 1:].reshape(-1)  # 目标是输入的偏移一位
        outputs = outputs.reshape(-1, outputs.shape[2])
        loss: Tensor = criterion(outputs, targets)

        # 反向传播
        [o.zero_grad() for o in optimizers]
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=cfg.grad_clip)
        if cfg.finetune_encoder:
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, encoder.parameters()),
                max_norm=cfg.grad_clip,
            )

        # 更新参数
        [o.step() for o in optimizers]

        with torch.no_grad():
            targets_not_padding = targets != 0
            num_targets: int = targets_not_padding.long().sum().item()
            num_correct: int = (
                torch.logical_and(outputs.argmax(-1) == targets, targets_not_padding)
                .long()
                .sum()
                .item()
            )

        loss_meter.update(loss.item(), num_targets)
        acc_meter.update(num_correct / num_targets, num_targets)
        pbar.set_postfix(
            {"Loss": f"{loss_meter.avg:.4f}", "Acc": f"{acc_meter.avg:.1%}"}
        )

    # 更新学习率
    [sched.step(loss_meter.avg) for sched in schedulers]

    return loss_meter.avg


def decode_caption(caption: list[int], rev_vocab: dict[int, str]) -> str:
    """Decode caption to sentence with vocabulary.

    Args:
        caption (list[int]): token ids.
        rev_vocab (dict[int, str]): map of token id to word.

    Returns:
        str: Decoded sentences.
    """
    words: list[str] = []
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
    """Evaluate the model (validation or test).

    Args:
        encoder (Encoder): Encoder model.
        decoder (Decoder): Decoder model.
        dataloader (DataLoader): DataLoader for evaluation.
        device (str): device name.
        vocab (dict[str, int]): map of token to token id.

    Returns:
        dict[str, float]: BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores.
    """
    device = cfg.device
    encoder.eval()
    decoder.eval()
    hypotheses: list[list[int]] = []
    references: list[list[list[int]]] = []

    for imgs, captions, lengths, all_captions in tqdm(dataloader, desc="Evaluating"):
        all_captions: list[list[list[int]]]
        imgs: Tensor
        # 移到设备上
        imgs = imgs.to(device)

        # Strip <start> and <end>
        preds = decoder.generate_caption(
            encoder(imgs), vocab, max_len=cfg.max_decode_len, beam_size=cfg.beam_size
        )
        for i, pred in enumerate(preds):
            preds[i] = pred[1:-1]
        for i, caps in enumerate(all_captions):
            for j, cap in enumerate(caps):
                caps[j] = cap[1:-1]
        hypotheses.extend(preds)
        references.extend(all_captions)

    print("Calculating BLEU score...")
    # bleu1 = corpus_bleu(references, hypotheses, weights=(1.0, 0, 0, 0))
    # bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
    # bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))

    return {
        # "bleu-1": bleu1,
        # "bleu-2": bleu2,
        # "bleu-3": bleu3,
        "bleu-4": bleu4,
    }


def main():
    """
    主函数：训练和评估模型
    """
    # 配置
    cfg = Config()
    if not torch.cuda.is_available():
        cfg.device = "cpu"
        print("Warning: Using CPU for training.")
    print("Config:")
    print(json.dumps(asdict(cfg), indent=2, ensure_ascii=False))

    timestamp = datetime.now().strftime(r"%Y%m%d-%H%M%S")
    output_dir = Path(cfg.output_dir) / timestamp
    print(f"Output dir: {output_dir}")

    # 训练数据增强
    train_transform = transforms.Compose(
        [
            # transforms.RandomResizedCrop(224),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 验证数据变换
    val_transform = transforms.Compose(
        [
            # transforms.Resize(224),
            # transforms.CenterCrop(224),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 创建数据集和数据加载器
    # Train images cannot be cached
    train_dataset = CaptionDataset(
        cfg.dataset_dir, split="train", transform=train_transform, cache=False
    )
    vocab = train_dataset.vocab
    vocab_size = len(vocab)
    val_dataset = CaptionDataset(
        cfg.dataset_dir, split="val", transform=val_transform, cache=True
    )
    print(
        f"Dataset: {train_dataset.split} {len(train_dataset):,} captions, "
        f"{val_dataset.split} {len(val_dataset):,} captions"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 创建模型
    device = cfg.device
    encoder, decoder = load_models(vocab_size, cfg)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore <pad>
    optimizers = [
        optim.Adam(
            decoder.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
    ]
    if cfg.finetune_encoder:
        optimizers.append(
            optim.Adam(
                filter(lambda p: p.requires_grad, encoder.parameters()),
                lr=cfg.encoder_learning_rate,
                weight_decay=1e-5,
            )
        )
    schedulers = [
        ReduceLROnPlateau(optimizer, factor=0.5, patience=5) for optimizer in optimizers
    ]

    # 训练循环
    best_bleu4 = -1.0

    for epoch in range(cfg.num_epochs):
        # 训练
        train_loss = train_one_epoch(
            epoch,
            encoder,
            decoder,
            train_loader,
            criterion,
            optimizers,
            schedulers,
            cfg,
        )
        print(f"Train loss: {train_loss:.4f}")
        # 验证
        scores = evaluate(encoder, decoder, val_loader, vocab, cfg)
        bleu4 = scores["bleu-4"]
        print(f"BLEU-4: {bleu4:.4f}")

        # 保存最佳模型
        if bleu4 > best_bleu4:
            best_bleu4 = bleu4
            if not output_dir.exists():
                output_dir.mkdir(parents=True, exist_ok=True)
                # Dump config
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
                f"Best model saved to: {output_dir / 'best_model.pt'}  (epoch: {epoch + 1}, BLEU-4: {best_bleu4:.4f})"
            )


if __name__ == "__main__":
    main()
