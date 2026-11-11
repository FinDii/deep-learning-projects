

# train_srresnet.py
# 独立训练脚本，使用 SRDataset 预训练 SRResNet

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from sr_dataset import SRDatasetConfig, SRDataset
from model import srresnet_x4
from utils import AverageMeter, ProgressMeter
from image_quality_assessment import PSNR, SSIM

# ============= 配置 =============
splits_info = {
    'train': {
        'image_folder': '/content/drive/MyDrive/datasets/DIV2K_HRLR/DIV2K_train_HR',
        'lr_folder': '/content/drive/MyDrive/datasets/DIV2K_HRLR/DIV2K_train_LR_bicubic/X4'
    },
    'valid': {
        'image_folder': '/content/drive/MyDrive/datasets/DIV2K_HRLR/DIV2K_valid_HR',
        'lr_folder': '/content/drive/MyDrive/datasets/DIV2K_HRLR/DIV2K_valid_LR_bicubic/X4'
    },
    'Set5': {
        'image_folder': '/content/drive/MyDrive/datasets/DIV2K_HRLR/benchmark/Set5/HR',
        'lr_folder': '/content/drive/MyDrive/datasets/DIV2K_HRLR/benchmark/Set5/LR_bicubic/X4'
    },
    'Set14': {
        'image_folder': '/content/drive/MyDrive/datasets/DIV2K_HRLR/benchmark/Set14/HR',
        'lr_folder': '/content/drive/MyDrive/datasets/DIV2K_HRLR/benchmark/Set14/LR_bicubic/X4'
    },
    'Urban100': {
        'image_folder': '/content/drive/MyDrive/datasets/DIV2K_HRLR/benchmark/Urban100/HR',
        'lr_folder': '/content/drive/MyDrive/datasets/DIV2K_HRLR/benchmark/Urban100/LR_bicubic/X4'
    },
    'BSDS100': {
        'image_folder': '/content/drive/MyDrive/datasets/DIV2K_HRLR/benchmark/B100/HR',
        'lr_folder': '/content/drive/MyDrive/datasets/DIV2K_HRLR/benchmark/B100/LR_bicubic/X4'
    },
}

# 随机种子
seed = 0
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g_train = torch.Generator()
g_train.manual_seed(seed)

g_valid = torch.Generator()
g_valid.manual_seed(seed + 1)

#  evaluate 函数：同时测试 Raw 和 EMA
def evaluate(eval_model, valid_loader, psnr_model, ssim_model, device):
    eval_model.eval()
    psnr_vals, ssim_vals = [], []
    with torch.no_grad():
        for lr_img, hr_img in valid_loader:
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)
            sr = eval_model(lr_img)
            psnr = psnr_model(sr, hr_img)
            ssim = ssim_model(sr, hr_img)
            psnr_vals.append(psnr.item())
            ssim_vals.append(ssim.item())
    avg_psnr = sum(psnr_vals) / len(psnr_vals)
    avg_ssim = sum(ssim_vals) / len(ssim_vals)
    return avg_psnr, avg_ssim

def main():
    csv_folder = '/content/drive/MyDrive/DIV2K_Results/output_csv'
    # 直接从 splits_info 字典读取路径
    train_hr_folder = splits_info['train']['image_folder']
    train_lr_folder = splits_info['train']['lr_folder']
    valid_hr_folder = splits_info['valid']['image_folder']
    valid_lr_folder = splits_info['valid']['lr_folder']

    # 固定随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    crop_size = 96
    scale = 4
    patch_num = 100
    batch_size = 16
    num_workers = 2
    epochs = 100    # 配合早停
    lr = 1e-4
    milestones = [40, 70, 90]
    gamma = 0.5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建训练集和验证集配置
    train_config = SRDatasetConfig(
        csv_folder=csv_folder,
        HR_data_folder=train_hr_folder,
        LR_data_folder=train_lr_folder,
        crop_size=crop_size,
        scaling_factor=scale,
        patch_num=patch_num,
        cache=True
    )
    valid_config = SRDatasetConfig(
        csv_folder=csv_folder,
        HR_data_folder=valid_hr_folder,
        LR_data_folder=valid_lr_folder,
        crop_size=-1,         # 验证时不裁剪
        scaling_factor=scale,
        patch_num=1,                # 验证时每个图像只取一次
        cache=True
    )

    # 数据集与 DataLoader
    train_dataset = SRDataset("train", train_config)
    valid_dataset = SRDataset("valid", valid_config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g_train
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,                # 验证时 batch_size=1 方便计算每张图的指标
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g_valid
    )

    # 模型、损失、优化器、调度器
    model = srresnet_x4().to(device)

    # EMA模型
    ema_decay = 0.999

    ema_model = AveragedModel(
        model,
        multi_avg_fn=get_ema_multi_avg_fn(ema_decay),
        use_buffers=True
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    scaler = GradScaler()

    # PSNR/SSIM 评估器（用于验证）
    psnr_model = PSNR(crop_border=scale, only_test_y_channel=True).to(device)
    ssim_model = SSIM(crop_border=scale, only_test_y_channel=True).to(device)

    # 最佳指标
    best_raw_psnr = 0.0
    best_ema_psnr = 0.0

    # 早停参数
    patience = 20          # 连续20个epoch不提升则停止
    counter = 0

    # 训练循环
    for epoch in range(epochs):
        # ----- 训练 -----
        model.train()
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':6.6f')
        progress = ProgressMeter(len(train_loader), [batch_time, data_time, losses],
                                 prefix=f"Epoch [{epoch+1}/{epochs}]")

        end = time.time()
        for i, (lr_img, hr_img) in enumerate(train_loader):
            lr_img = lr_img.to(device, non_blocking=True)
            hr_img = hr_img.to(device, non_blocking=True)

            data_time.update(time.time() - end)

            optimizer.zero_grad()
            with autocast():
                sr = model(lr_img)
                loss = criterion(sr, hr_img)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 更新 EMA
            ema_model.update_parameters(model)

            losses.update(loss.item(), lr_img.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0:
                progress.display(i)

        scheduler.step()

        # 验证
        raw_psnr, raw_ssim = evaluate(model, valid_loader, psnr_model, ssim_model, device)
        ema_psnr, ema_ssim = evaluate(ema_model, valid_loader, psnr_model, ssim_model, device)
        print(f"[Epoch {epoch+1}] RAW: PSNR={raw_psnr:.2f}, SSIM={raw_ssim:.4f} | EMA: PSNR={ema_psnr:.2f}, SSIM={ema_ssim:.4f}")

        # 判断本 epoch 是否有提升
        raw_improved = raw_psnr > best_raw_psnr
        ema_improved = ema_psnr > best_ema_psnr
        
        # 保存最佳 RAW
        if raw_improved:
            best_raw_psnr = raw_psnr
            torch.save(
                model.state_dict(),
                "/content/drive/MyDrive/DIV2K_Results/srresnet_x4_best_raw.pth"
            )
            print(f"  -> Best RAW model saved (PSNR: {best_raw_psnr:.2f})")
        
        # 保存最佳 EMA
        if ema_improved:
            best_ema_psnr = ema_psnr
            torch.save(
                ema_model.module.state_dict(),
                "/content/drive/MyDrive/DIV2K_Results/srresnet_x4_best_ema.pth"
            )
            print(f"  -> Best EMA model saved (PSNR: {best_ema_psnr:.2f})")
        
        # Early stopping：只有 RAW 和 EMA 都没有提升才计数
        if raw_improved or ema_improved:
            counter = 0
        else:
            counter += 1
            print(f"  -> No improvement for {counter} epochs.")
        
        if counter >= patience:
            print(
                f"Early stopping triggered at epoch {epoch+1}. "
                f"Best RAW PSNR: {best_raw_psnr:.2f}, "
                f"Best EMA PSNR: {best_ema_psnr:.2f}"
            )
            break

    print("Training finished!")

if __name__ == "__main__":
    main()