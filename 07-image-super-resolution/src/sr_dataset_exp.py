
import csv
import os
import random
from dataclasses import dataclass

import torchvision.transforms.functional as TF
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import decode_image
from tqdm import tqdm


@dataclass
class SRDatasetConfig:
    csv_folder: str = "/content/drive/MyDrive/DIV2K_Results/output_csv"
    HR_data_folder: str = '/content/drive/MyDrive/datasets/DIV2K_HRLR/DIV2K_train_HR'
    LR_data_folder: str = '/content/drive/MyDrive/datasets/DIV2K_HRLR/DIV2K_train_LR_bicubic/X4'
    crop_size: int = 96
    scaling_factor: int = 4
    patch_num: int = 100
    cache: bool = True
    color_augment: bool = False
    noise_augment: bool = False  # 是否开启高级增强


class SRDataset(Dataset):
    def __init__(self, split: str, config: SRDatasetConfig):
        self.cache = config.cache
        self.split = split
        self.csv_folder = config.csv_folder
        self.HR_data_folder = config.HR_data_folder
        self.LR_data_folder = config.LR_data_folder
        self.crop_size = int(config.crop_size)
        self.scaling_factor = int(config.scaling_factor)
        self.patch_num = config.patch_num
        self.color_augment = config.color_augment
        self.noise_augment = config.noise_augment

        assert self.split in {"train", "valid", "Set5", "Set14", "B100", "Urban100", "BSDS100"}

        self.HR_images = []
        self.LR_images = []
        with open(os.path.join(self.csv_folder, self.split + "_images.csv"), "r") as f:
            reader = csv.reader(f)
            for line in reader:
                self.HR_images.append(line[0])
                if os.path.exists(os.path.join(self.LR_data_folder, line[1])):
                    self.LR_images.append(line[1])
                else:
                    assert os.path.exists(os.path.join(self.LR_data_folder, line[0]))
                    self.LR_images.append(line[0])
        if self.cache:
            self.images_cache = []
            for i in tqdm(range(len(self.HR_images)), desc="Loading dataset"):
                # Load HR and LR images using torchvision
                hr_path = os.path.join(self.HR_data_folder, self.HR_images[i])
                lr_path = os.path.join(self.LR_data_folder, self.LR_images[i])

                hr_img = (
                    decode_image(hr_path, mode="RGB").float() / 255.0
                )  # C x H x W, float32
                lr_img = decode_image(lr_path, mode="RGB").float() / 255.0

                # Ensure size consistency
                hr_img, lr_img = self._center_crop_and_resize_if_needed(hr_img, lr_img)
                self.images_cache.append((hr_img, lr_img))

    def _center_crop_and_resize_if_needed(self, hr_img: Tensor, lr_img: Tensor):
        # Only if HR and LR shapes are inconsistent
        C, hr_h, hr_w = hr_img.shape
        C2, lr_h, lr_w = lr_img.shape
        assert C == C2

        expected_lr_h = hr_h // self.scaling_factor
        expected_lr_w = hr_w // self.scaling_factor

        if lr_h != expected_lr_h or lr_w != expected_lr_w:
            # Center crop HR to be divisible by scaling factor
            new_hr_h = (hr_h // self.scaling_factor) * self.scaling_factor
            new_hr_w = (hr_w // self.scaling_factor) * self.scaling_factor
            top = (hr_h - new_hr_h) // 2
            left = (hr_w - new_hr_w) // 2
            hr_img = TF.crop(hr_img, top, left, new_hr_h, new_hr_w)

            # Resize LR image from HR
            lr_img = TF.resize(
                hr_img,
                [new_hr_h // self.scaling_factor, new_hr_w // self.scaling_factor],
                interpolation=TF.InterpolationMode.BICUBIC,
            )
        return hr_img, lr_img

    def __getitem__(self, i: int):
        if self.split == "train":
            i = i // self.patch_num

        if self.cache:
            hr_img, lr_img = self.images_cache[i]
        else:
            # Load HR and LR images using torchvision
            hr_path = os.path.join(self.HR_data_folder, self.HR_images[i])
            lr_path = os.path.join(self.LR_data_folder, self.LR_images[i])

            hr_img = (
                decode_image(hr_path, mode="RGB").float() / 255.0
            )  # C x H x W, float32
            lr_img = decode_image(lr_path, mode="RGB").float() / 255.0

            # Ensure size consistency
            hr_img, lr_img = self._center_crop_and_resize_if_needed(hr_img, lr_img)

        if self.split == "train":
            lr_img, hr_img = self._sample_patch(lr_img, hr_img)
            lr_img, hr_img = self._augment(lr_img, hr_img)

        return lr_img, hr_img

    def _sample_patch(self, lr_img: Tensor, hr_img: Tensor):
        _, h, w = lr_img.shape
        lr_crop_size = self.crop_size // self.scaling_factor
        left = random.randint(2, w - lr_crop_size - 2)
        top = random.randint(2, h - lr_crop_size - 2)

        lr_patch = TF.crop(lr_img, top, left, lr_crop_size, lr_crop_size)
        hr_patch = TF.crop(
            hr_img,
            top * self.scaling_factor,
            left * self.scaling_factor,
            self.crop_size,
            self.crop_size,
        )
        return lr_patch, hr_patch

    def _augment(self, lr_img: Tensor, hr_img: Tensor):
        if random.random() < 0.5:
            lr_img = TF.hflip(lr_img)
            hr_img = TF.hflip(hr_img)
        if random.random() < 0.5:
            lr_img = TF.vflip(lr_img)
            hr_img = TF.vflip(hr_img)
        if random.random() < 0.5:
            angle = 90
            lr_img = TF.rotate(lr_img, angle)
            hr_img = TF.rotate(hr_img, angle)

        # 新增颜色抖动（亮度、对比度、饱和度）
        # 颜色增强（仅当 color_augment=True 时启用）
        if self.color_augment:
            if random.random() < 0.5:
                brightness_factor = random.uniform(0.8, 1.2)
                contrast_factor = random.uniform(0.8, 1.2)
                saturation_factor = random.uniform(0.8, 1.2)
                lr_img = TF.adjust_brightness(lr_img, brightness_factor)
                hr_img = TF.adjust_brightness(hr_img, brightness_factor)
                lr_img = TF.adjust_contrast(lr_img, contrast_factor)
                hr_img = TF.adjust_contrast(hr_img, contrast_factor)
                lr_img = TF.adjust_saturation(lr_img, saturation_factor)
                hr_img = TF.adjust_saturation(hr_img, saturation_factor)

        if self.noise_augment:
            # 高斯噪声（仅加在 LR 上）
            if random.random() < 0.3:
                noise = torch.randn_like(lr_img) * 0.02
                lr_img = lr_img + noise
                lr_img = torch.clamp(lr_img, 0.0, 1.0)

        return lr_img, hr_img

    def __len__(self):

        if self.split == "train":
            return len(self.HR_images) * self.patch_num
        else:
            return len(self.HR_images)