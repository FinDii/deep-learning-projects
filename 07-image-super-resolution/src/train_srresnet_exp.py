
import os
import time
import argparse
import random
import numpy as np
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

import sys
sys.path.append('/content/SRGAN-PyTorch')

from sr_dataset_exp import SRDatasetConfig, SRDataset
from model import srresnet_x4
from utils import AverageMeter, ProgressMeter
from image_quality_assessment import PSNR, SSIM

# ============= 配置 =============
# 随机种子
seed = 0

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

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
    parser = argparse.ArgumentParser(description='Train SRResNet')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--channels', type=int, default=64, help='Number of channels in model')
    parser.add_argument('--epochs', type=int, default=100, help='Total epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--crop_size', type=int, default=96, help='Crop size for training')
    parser.add_argument('--patch_num', type=int, default=100, help='Patches per image per epoch')
    parser.add_argument('--milestones', type=int, nargs='+', default=[40, 70, 90], help='LR decay milestones')
    parser.add_argument('--gamma', type=float, default=0.5, help='LR decay factor')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--save_dir', type=str, default='/content/drive/MyDrive/DIV2K_Results', help='Directory to save models')
    parser.add_argument('--csv_folder', type=str, default='/content/drive/MyDrive/DIV2K_Results/output_csv', help='CSV folder')
    parser.add_argument('--run_name', type=str, default='srresnet_x4_exp', help='Name of this experiment')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of data loading workers')
    parser.add_argument('--cache', action='store_true', help='Enable dataset caching')
    parser.add_argument('--color_augment', action='store_true', help='Enable color augmentation')
    parser.add_argument('--noise_augment', action='store_true', help='Add Gaussian noise to LR')
    args = parser.parse_args()

    # 随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    g_train = torch.Generator()
    g_train.manual_seed(seed)

    g_valid = torch.Generator()
    g_valid.manual_seed(seed + 1)

    # 提取参数
    lr = args.lr
    weight_decay = args.weight_decay
    channels = args.channels
    epochs = args.epochs
    batch_size = args.batch_size
    crop_size = args.crop_size
    patch_num = args.patch_num
    milestones = args.milestones
    gamma = args.gamma
    patience = args.patience
    save_dir = args.save_dir
    run_name = args.run_name
    os.makedirs(save_dir, exist_ok=True)
    csv_folder = args.csv_folder
    num_workers = args.num_workers
    cache = args.cache
    color_augment = args.color_augment
    noise_augment = args.noise_augment
    scale = 4  # 定义 scale

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 路径
    train_hr_folder = splits_info['train']['image_folder']
    train_lr_folder = splits_info['train']['lr_folder']
    valid_hr_folder = splits_info['valid']['image_folder']
    valid_lr_folder = splits_info['valid']['lr_folder']

    # 创建训练集和验证集配置
    train_config = SRDatasetConfig(
        csv_folder=csv_folder,
        HR_data_folder=train_hr_folder,
        LR_data_folder=train_lr_folder,
        crop_size=crop_size,
        scaling_factor=scale,
        patch_num=patch_num,
        cache=cache,
        color_augment=color_augment,
        noise_augment=noise_augment
    )
    valid_config = SRDatasetConfig(
        csv_folder=csv_folder,
        HR_data_folder=valid_hr_folder,
        LR_data_folder=valid_lr_folder,
        crop_size=-1,         # 验证时不裁剪
        scaling_factor=scale,
        patch_num=1,                # 验证时每个图像只取一次
        cache=cache,
        color_augment=False,
        noise_augment=False         # 验证集不高级增强
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
    model = srresnet_x4(channels=channels).to(device)

    ema_decay = 0.999
    ema_model = AveragedModel(
        model,
        multi_avg_fn=get_ema_multi_avg_fn(ema_decay),
        use_buffers=True
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    scaler = GradScaler()

    # PSNR/SSIM 评估器（用于验证）
    psnr_model = PSNR(crop_border=scale, only_test_y_channel=True).to(device)
    ssim_model = SSIM(crop_border=scale, only_test_y_channel=True).to(device)

    # 最佳指标
    best_raw_psnr = 0.0
    best_ema_psnr = 0.0
    raw_save_path = os.path.join(save_dir, f'{run_name}_best_raw.pth')
    ema_save_path = os.path.join(save_dir, f'{run_name}_best_ema.pth')
    # 记录csv
    log_path = os.path.join(save_dir, f'{run_name}_metrics.csv')
    with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch',
                'raw_psnr',
                'raw_ssim',
                'ema_psnr',
                'ema_ssim',
                'lr'
            ])

    # 早停指标
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

        # 验证
        raw_psnr, raw_ssim = evaluate(model, valid_loader, psnr_model, ssim_model, device)
        ema_psnr, ema_ssim = evaluate(ema_model, valid_loader, psnr_model, ssim_model, device)
        print(f"[Epoch {epoch+1}] RAW: PSNR={raw_psnr:.2f}, SSIM={raw_ssim:.4f} | EMA: PSNR={ema_psnr:.2f}, SSIM={ema_ssim:.4f}")

        # 在每次验证后记录指标
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                raw_psnr,
                raw_ssim,
                ema_psnr,
                ema_ssim,
                optimizer.param_groups[0]['lr']
            ])

        # 早停和保存
        raw_improved = raw_psnr > best_raw_psnr
        ema_improved = ema_psnr > best_ema_psnr

        if raw_improved:
            best_raw_psnr = raw_psnr
            torch.save(model.state_dict(), raw_save_path)
            print(f"  -> Best RAW model saved (PSNR: {best_raw_psnr:.2f})")

        if ema_improved:
            best_ema_psnr = ema_psnr
            torch.save(ema_model.module.state_dict(), ema_save_path)
            print(f"  -> Best EMA model saved (PSNR: {best_ema_psnr:.2f})")

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

        scheduler.step()

    print("Training finished!")

if __name__ == "__main__":
    main()