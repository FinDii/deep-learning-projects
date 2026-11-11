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

    Args:
        batch: List of tuples from CaptionDataset.__getitem__
            For train split: [(img1, caption1), (img2, caption2), ...]
            For val/test split: [(img1, caption1, all_captions1), (img2, caption2, all_captions2), ...]

    Returns:
        For train split:
            imgs: Tensor of shape (batch_size, channels, height, width)
            captions: Tensor of shape (batch_size, max_caption_length)
            lengths: Tensor of shape (batch_size,) containing lengths of each caption
        For val/test split:
            imgs: Tensor of shape (batch_size, channels, height, width)
            captions: Tensor of shape (batch_size, max_caption_length)
            lengths: Tensor of shape (batch_size,) containing lengths of each caption
            all_captions: List of all captions for each image in the batch
    """
    # 检查是否是训练集
    is_train = len(batch[0]) == 2

    if is_train:
        # 训练集: (img, caption)
        imgs, captions = zip(*batch)
    else:
        # 验证/测试集: (img, caption, all_captions)
        imgs, captions, all_captions = zip(*batch)

    # Process imgs
    if isinstance(imgs[0], Tensor):
        imgs = torch.stack(imgs, 0)

    # Process captions
    lengths = [len(cap) for cap in captions]
    # Find maximum length
    max_length = max(lengths)
    # Pad captions
    padded_captions = torch.full(
        (len(captions), max_length), PAD_TOKEN, dtype=torch.long
    )
    for i, cap in enumerate(captions):
        end = lengths[i]
        padded_captions[i, :end] = torch.tensor(cap[:end], dtype=torch.long)
    # Create lengths tensor
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
        split: DatasetSplit = "train",
        transform=None,
        cache=True,
        max_cap_len=-1,
    ):
        """Load dataset.

        Args:
            root (str | Path): Dataset root.
            split (str): Dataset split, options: 'train', 'val', 'test'. Default: 'train'.
            transform (Any | None): Input transform. Default: None.
            cache (bool): If set, images are cached for faster loading.
            max_cap_len (int): Maximum caption length. Default: -1 (disabled).
        """
        super().__init__()
        self.split = split
        assert self.split in typing.get_args(DatasetSplit)

        root = Path(root)
        self.root = root
        if not os.path.isdir(root):
            raise RuntimeError(f"Caption dataset does not exist in {root}")

        self.img_dir = root / "Images"
        if not os.path.isdir(self.img_dir):
            self.img_dir = root / "images"

        # Load captions
        with open(root / "captions_encoded.json", "r") as f:
            captions: list[dict] = json.load(f)
        with open(root / "vocab.json", "r") as f:
            self.vocab: dict[str, int] = json.load(f)
        # Reverse vocabulary: index -> word
        self.rev_vocab: dict[int, str] = {i: w for w, i in self.vocab.items()}
        # Flatten captions
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
        self.cpi = len(captions[0]["sentences"])
        """Captions per image."""
        self.transform = transform

        # Construct image cache
        self.img_cache = []
        if cache:
            for name in tqdm(self.img_names, desc=f"Caching {self.split} images"):
                img = Image.open(self.img_dir / name)
                if self.transform:
                    img = self.transform(img)
                self.img_cache.append(img)

    def __getitem__(self, idx: int):
        if self.img_cache:
            img = self.img_cache[self.captions[idx]["img_idx"]]
        else:
            name = self.img_names[self.captions[idx]["img_idx"]]
            img = Image.open(self.img_dir / name)
            if self.transform:
                img = self.transform(img)
        caption: list[int] = self.captions[idx]["caption"]

        if self.split == "train":
            return img, caption
        else:
            # 对于验证或测试，还返回每张图像的所有描述以计算BLEU-4分数
            start_idx = (idx // self.cpi) * self.cpi
            end_idx = start_idx + self.cpi
            all_captions = [cap["caption"] for cap in self.captions[start_idx:end_idx]]
            return img, caption, all_captions

    def __len__(self):
        return len(self.captions)
