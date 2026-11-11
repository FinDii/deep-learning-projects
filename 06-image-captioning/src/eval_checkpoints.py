# %%writefile eval_checkpoints.py

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms as transforms
from nltk.translate.bleu_score import corpus_bleu
from torch.utils.data import DataLoader
from tqdm import tqdm

from caption_dataset import CaptionDataset, collate_fn
from models import Encoder, RNNDecoder, RNNDecoderWithAttention

# 改最佳模型路径
ckpt_paths = [
    Path("/kaggle/input/models/fdiiiiii/flickr8k-best-models/pytorch/default/1/Baseline_best_model.pt"), #baseline
    Path("/kaggle/input/models/fdiiiiii/flickr8k-best-models/pytorch/default/1/SR_best_model.pt"), # stronger regulation
    Path("/kaggle/input/models/fdiiiiii/flickr8k-best-models/pytorch/default/1/Att_best_model.pt"), #attention
    Path("/kaggle/input/models/fdiiiiii/flickr8k-best-models/pytorch/default/1/ASS_best_model.pt"), #attetion+scheduled sampling
    Path("/kaggle/input/models/fdiiiiii/flickr8k-best-models/pytorch/default/1/finetune_best_model.pt"), #finetune
]


@torch.no_grad()
def evaluate_model(encoder, decoder, dataloader, vocab, device, beam_size=1, max_len=30):
    encoder.eval()
    decoder.eval()

    hypotheses = []
    references = []

    for imgs, captions, lengths, all_captions in tqdm(
        dataloader, desc=f"Evaluating beam={beam_size}"
    ):
        imgs = imgs.to(device)

        features = encoder(imgs)
        preds = decoder.generate_caption(
            features,
            vocab,
            max_len=max_len,
            beam_size=beam_size,
        )

        # remove <start> and <end>
        cleaned_preds = []
        for pred in preds:
            cleaned_preds.append(pred[1:-1])

        cleaned_refs = []
        for caps in all_captions:
            cleaned_caps = []
            for cap in caps:
                cleaned_caps.append(cap[1:-1])
            cleaned_refs.append(cleaned_caps)

        hypotheses.extend(cleaned_preds)
        references.extend(cleaned_refs)

    bleu4 = corpus_bleu(
        references,
        hypotheses,
        weights=(0.25, 0.25, 0.25, 0.25),
    )
    return bleu4


def load_checkpoint_model(ckpt_path, vocab_size, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]

    finetune_encoder = cfg.get("finetune_encoder", False)
    use_attention = cfg.get("use_attention", False)

    encoder = Encoder(encoded_size=(7, 7), finetune=finetune_encoder).to(device)

    # Probe feature dimension
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224, device=device)
        feature_dim = encoder(dummy).shape[-1]

    if use_attention:
        decoder = RNNDecoderWithAttention(
            cfg.get("embed_dim", 512),
            cfg.get("hidden_dim", 1024),
            vocab_size,
            feature_dim,
            dropout=cfg.get("dropout", 0.2),
        ).to(device)
    else:
        decoder = RNNDecoder(
            cfg.get("embed_dim", 512),
            cfg.get("hidden_dim", 1024),
            vocab_size,
            feature_dim,
            dropout=cfg.get("dropout", 0.2),
        ).to(device)

    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])

    return encoder, decoder, cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="/kaggle/input/datasets/adityajn105/flickr8k")
    parser.add_argument("--preprocess_dir", type=str, default="/kaggle/input/datasets/fdiiiiii/flickr8k")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--beam_sizes", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_len", type=int, default=30)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    device = args.device

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    dataset = CaptionDataset(
        args.dataset_dir,
        preprocess_dir=args.preprocess_dir,
        split=args.split,
        transform=transform,
        cache=False,
    )

    vocab = dataset.vocab
    vocab_size = len(vocab)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # ckpt_paths = sorted(Path(args.output_dir).glob("*/best_model.pt"))

    if len(ckpt_paths) == 0:
        raise FileNotFoundError(f"No best_model.pt found under {args.output_dir}")

    rows = []

    for ckpt_path in ckpt_paths:
        print("\n" + "=" * 80)
        print(f"Loading checkpoint: {ckpt_path}")

        encoder, decoder, cfg = load_checkpoint_model(ckpt_path, vocab_size, device)

        model_name = ckpt_path.parent.name
        use_attention = cfg.get("use_attention", False)
        use_scheduled_sampling = cfg.get("use_scheduled_sampling", 0)
        finetune_encoder = cfg.get("finetune_encoder", False)
        lr = cfg.get("learning_rate", None)
        dropout = cfg.get("dropout", None)

        for beam_size in args.beam_sizes:
            bleu4 = evaluate_model(
                encoder,
                decoder,
                dataloader,
                vocab,
                device,
                beam_size=beam_size,
                max_len=args.max_len,
            )

            row = {
                "checkpoint": model_name,
                "path": str(ckpt_path),
                "split": args.split,
                "beam_size": beam_size,
                "bleu4": bleu4,
                "use_attention": use_attention,
                "use_scheduled_sampling": use_scheduled_sampling,
                "finetune_encoder": finetune_encoder,
                "learning_rate": lr,
                "dropout": dropout,
            }
            rows.append(row)

            print(
                f"{model_name} | attention={use_attention} | "
                f"scheduled_sampling={use_scheduled_sampling} | "
                f"beam={beam_size} | BLEU-4={bleu4:.4f}"
            )

    df = pd.DataFrame(rows)
    
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"beam_search_comparison_{args.split}.csv"
    df.to_csv(save_path, index=False)

    print("\nFinal results:")
    print(df.sort_values("bleu4", ascending=False))
    print(f"\nSaved to: {save_path}")


if __name__ == "__main__":
    main()