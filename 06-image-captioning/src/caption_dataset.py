# %%writefile caption_dataset.py

import json
import os
import typing
from pathlib import Path
from typing import Literal

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

PAD_TOKEN = 0
DatasetSplit = Literal["train", "val", "test"]


def collate_fn(batch: list[tuple]):
    """
    Collate function for CaptionDataset.
    """
    is_train = len(batch[0]) == 2

    if is_train:
        imgs, captions = zip(*batch)
    else:
        imgs, captions, all_captions = zip(*batch)

    if isinstance(imgs[0], Tensor):
        imgs = torch.stack(imgs, 0)

    lengths = [len(cap) for cap in captions]
    max_length = max(lengths)
    padded_captions = torch.full(
        (len(captions), max_length), PAD_TOKEN, dtype=torch.long
    )
    for i, cap in enumerate(captions):
        end = lengths[i]
        padded_captions[i, :end] = torch.tensor(cap[:end], dtype=torch.long)
    lengths = torch.tensor(lengths)

    if is_train:
        return imgs, padded_captions, lengths
    else:
        return imgs, padded_captions, lengths, all_captions


class CaptionDataset(Dataset):
    """Load preprocessed caption dataset."""

    def __init__(
        self,
        root: str | Path,
        preprocess_dir: str | Path, 
        split: DatasetSplit = "train",
        transform=None,
        cache=False,
        max_cap_len=-1,
    ):
        super().__init__()
        self.split = split
        assert self.split in typing.get_args(DatasetSplit)

        root = Path(root)
        preprocess_dir = Path(preprocess_dir) 

        self.root = root
        self.preprocess_dir = preprocess_dir

        if not root.is_dir():
            raise RuntimeError(f"Image root does not exist: {root}")

        self.img_dir = root / "Images"
        if not self.img_dir.is_dir():
            self.img_dir = root / "images"

        # 加载预处理文件
        enc_path = preprocess_dir / "captions_encoded.json"
        vocab_path = preprocess_dir / "vocab.json"
        if not enc_path.exists():
            raise FileNotFoundError(f"Missing {enc_path}")
        if not vocab_path.exists():
            raise FileNotFoundError(f"Missing {vocab_path}")

        with open(enc_path, "r") as f:
            captions: list[dict] = json.load(f)
        with open(vocab_path, "r") as f:
            self.vocab: dict[str, int] = json.load(f)

        self.rev_vocab: dict[int, str] = {i: w for w, i in self.vocab.items()}

        # 展平数据
        self.img_names: list[str] = []
        self.captions: list[dict] = []
        max_cap_len_in_data = 0
        for caption in captions:
            if caption["split"] != self.split:
                continue
            img_idx = len(self.img_names)
            self.img_names.append(caption["filename"])
            for sentence in caption["sentences"]:
                max_cap_len_in_data = max(max_cap_len_in_data, len(sentence))
                self.captions.append({"img_idx": img_idx, "caption": sentence})
        self.max_cap_len = max_cap_len_in_data if max_cap_len <= 0 else max_cap_len
        self.cpi = len(captions[0]["sentences"]) if captions else 5
        self.transform = transform

        # 图像缓存（如果 cache=True，但 preprocess_dir 可能是只读, 所以cache=False）
        self.img_cache = []
        if cache:
            cache_dir = preprocess_dir / "cache" / self.split
            cache_dir.mkdir(parents=True, exist_ok=True)
            for name in tqdm(self.img_names, desc=f"Caching {self.split} images"):
                img = Image.open(self.img_dir / name).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                self.img_cache.append(img)

    def __getitem__(self, idx: int):
        if self.img_cache:
            img = self.img_cache[self.captions[idx]["img_idx"]]
        else:
            img = Image.open(self.img_dir / self.img_names[self.captions[idx]["img_idx"]]).convert("RGB")
            if self.transform:
                img = self.transform(img)
        caption: list[int] = self.captions[idx]["caption"]

        if self.split == "train":
            return img, caption
        else:
            start_idx = (idx // self.cpi) * self.cpi
            end_idx = start_idx + self.cpi
            all_captions = [cap["caption"] for cap in self.captions[start_idx:end_idx]]
            return img, caption, all_captions

    def __len__(self):
        return len(self.captions)