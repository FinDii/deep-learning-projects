
# 基于预训练 SRResNet 的 SRGAN 训练脚本
# 兼容不同scale
# 增加实验改进参数

import os
import sys
import time
import argparse
import random
import csv
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter

sys.path.append('/content/SRGAN-PyTorch')

from sr_dataset_exp_scale import SRDatasetConfig, SRDataset
from model_srgan import SRResNet, discriminator_for_vgg, ContentLoss
from utils import AverageMeter, ProgressMeter
from image_quality_assessment import PSNR, SSIM

# 平滑版 L1
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        return torch.mean(
            torch.sqrt(
                (x - y) ** 2 + self.eps ** 2
            )
        )

# ============= 路径配置 =============
splits_info_multi = {
    'train': {
        'hr_folder': '/content/drive/MyDrive/datasets/DIV2K_HRLR/DIV2K_train_HR',
        'lr_folders': {
            '2': '/content/drive/MyDrive/datasets/DIV2K_HRLR/DIV2K_train_LR_bicubic/X2',
            '3': '/content/drive/MyDrive/datasets/DIV2K_HRLR/DIV2K_train_LR_bicubic/X3',
            '4': '/content/drive/MyDrive/datasets/DIV2K_HRLR/DIV2K_train_LR_bicubic/X4',
        }
    },

    'valid': {
        'hr_folder': '/content/drive/MyDrive/datasets/DIV2K_HRLR/DIV2K_valid_HR',
        'lr_folders': {
            '2': '/content/drive/MyDrive/datasets/DIV2K_HRLR/DIV2K_valid_LR_bicubic/X2',
            '3': '/content/drive/MyDrive/datasets/DIV2K_HRLR/DIV2K_valid_LR_bicubic/X3',
            '4': '/content/drive/MyDrive/datasets/DIV2K_HRLR/DIV2K_valid_LR_bicubic/X4',
        }
    }
}


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_hr_folder(split_info, scale):
    """
    Select scale-specific cropped HR folder if it exists.
    Otherwise use the original HR folder.
    """
    hr_base = split_info['hr_folder']
    hr_cropped_dir = os.path.join(hr_base, 'HR_cropped', f'X{scale}')

    if os.path.isdir(hr_cropped_dir):
        return hr_cropped_dir

    return hr_base

def evaluate(eval_model, valid_loader, psnr_model, ssim_model, device):
    """
    Evaluate average PSNR / SSIM over validation set.
    SR and HR must be strictly aligned.
    """
    eval_model.eval()

    psnr_vals = []
    ssim_vals = []

    with torch.no_grad():

        for lr_img, hr_img in valid_loader:

            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)

            sr = eval_model(lr_img)

            # Do NOT silently crop mismatched images.
            # Data preprocessing should guarantee exact alignment.
            if sr.shape != hr_img.shape:
                raise ValueError(
                    f"SR/HR size mismatch: "
                    f"SR={tuple(sr.shape)}, "
                    f"HR={tuple(hr_img.shape)}"
                )

            psnr_vals.append(psnr_model(sr, hr_img).item())
            ssim_vals.append(ssim_model(sr, hr_img).item())

    return (np.mean(psnr_vals), np.mean(ssim_vals))

def main():
    parser = argparse.ArgumentParser(description='Train SRGAN')
    # 训练超参数
    parser.add_argument('--g_lr', type=float, default=1e-4)
    parser.add_argument('--d_lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--channels', type=int, default=64, help='Number of channels in G')
    parser.add_argument('--scale', type=int, default=4, choices=[2, 3, 4], help='Super-resolution scale factor')
    parser.add_argument('--epochs', type=int, default=100, help='Total epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--crop_size', type=int, default=96, help='Crop size for training')
    parser.add_argument('--patch_num', type=int, default=100, help='Patches per image per epoch')
    parser.add_argument('--milestones', type=int, nargs='+', default=[40, 70, 90], help='LR decay milestones')
    parser.add_argument('--gamma', type=float, default=0.5, help='LR decay factor')

    # One-sided Label Smoothing
    parser.add_argument('--real_label', type=float, default=1.0, help='Real target label for discriminator; try 0.9 for one-sided label smoothing')

    # 让 adversarial component 前几 epoch 从 0 慢慢增加到目标值
    parser.add_argument('--adv_warmup_epochs', type=int, default=0, help='Linearly ramp adversarial weight from 0 to target value')

    # 损失权重
    # GitHub SRGAN x4 config baseline
    parser.add_argument('--pixel_weight', type=float, default=1.0, help='Weight for pixel loss')
    parser.add_argument('--content_weight', type=float, default=1.0, help='Weight for perceptual loss')
    parser.add_argument('--adversarial_weight', type=float, default=1e-3, help='Weight for adversarial loss')

    # 损失函数
    parser.add_argument('--pixel_loss', type=str, default='mse', choices=['mse', 'l1', 'charbonnier'], help='Pixel reconstruction loss')

    # 数据/路径
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save models. If omitted, automatically uses output_x{scale}.')
    parser.add_argument('--checkpoint_interval', type=int, default=20, help='Save periodic RAW/EMA checkpoints every N epochs. Use 0 to disable.')
    parser.add_argument('--csv_folder', type=str, default='/content/drive/MyDrive/DIV2K_Results/output_csv', help='CSV folder')
    parser.add_argument('--run_name', type=str, required=True, help='Unique experiment name, e.g. M1_l1')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of data loading workers')
    parser.add_argument('--cache', action='store_true', help='Enable dataset caching')
    parser.add_argument('--color_augment', action='store_true', help='Enable color augmentation')
    parser.add_argument('--noise_augment', action='store_true', help='Add Gaussian noise to LR')

    # 预训练模型
    parser.add_argument(
        '--pretrained_g',
        type=str,
        default=None,
        help='Path to scale-matched pretrained SRResNet generator'
    )
    parser.add_argument('--no_pretrained', action='store_true', help='Train G from scratch')
    args = parser.parse_args()

    # 参数 sanity checks
    # discriminator_for_vgg 硬编码要求 96×96 输入
    if args.crop_size != 96:
        raise ValueError(
            "discriminator_for_vgg requires crop_size=96"
        )

    if not 0.0 < args.real_label <= 1.0:
        raise ValueError(
            "--real_label must be in (0, 1]."
        )

    if args.adv_warmup_epochs < 0:
        raise ValueError(
            "--adv_warmup_epochs must be >= 0."
        )

    if args.g_lr <= 0 or args.d_lr <= 0:
        raise ValueError(
            "Learning rates must be > 0."
        )

    if args.adversarial_weight < 0:
        raise ValueError(
            "--adversarial_weight must be >= 0."
        )

    # 固定随机种子
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scale = args.scale

    # 自动输出目录
    if args.save_dir is None:
        save_dir = os.path.join(
            "/content/drive/MyDrive/DIV2K_Results",
            f"output_improve_x{scale}"
         )
    else:
        save_dir = args.save_dir

    os.makedirs(
        save_dir,
        exist_ok=True
    )

    print(f"Scale: x{scale}")
    print(f"Results will be saved to: {save_dir}")

    # 数据路径
    train_hr = get_hr_folder(splits_info_multi['train'], scale)
    train_lr = splits_info_multi['train']['lr_folders'][str(scale)]
    valid_hr = get_hr_folder(splits_info_multi['valid'], scale)
    valid_lr = splits_info_multi['valid']['lr_folders'][str(scale)]

    # 训练集配置
    train_config = SRDatasetConfig(
        csv_folder=args.csv_folder,
        HR_data_folder=train_hr,
        LR_data_folder=train_lr,
        crop_size=args.crop_size,
        scaling_factor=scale,
        patch_num=args.patch_num,
        cache=args.cache,
        scale=scale,
        color_augment=args.color_augment,
        noise_augment=args.noise_augment
    )
    valid_config = SRDatasetConfig(
        csv_folder=args.csv_folder,
        HR_data_folder=valid_hr,
        LR_data_folder=valid_lr,
        crop_size=-1,
        scaling_factor=scale,
        patch_num=1,
        cache=args.cache,
        scale=scale,
        color_augment=False,
        noise_augment=False
    )

    train_dataset = SRDataset("train", train_config)
    valid_dataset = SRDataset("valid", valid_config)

    g_train = torch.Generator()
    g_train.manual_seed(seed)
    g_valid = torch.Generator()
    g_valid.manual_seed(seed + 1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g_train
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g_valid
    )

    # ===== 1. 构建生成器 (G) =====
    g_model = SRResNet(channels=args.channels, upscale=scale).to(device)

    # 加载预训练权重
    if args.no_pretrained:
        print(f"Training x{scale} generator from scratch.")
    else:
        if args.pretrained_g is None:
            raise ValueError(
                f"--pretrained_g must be provided "
                f"for SRGAN x{scale}"
            )

        if not os.path.exists(args.pretrained_g):
            raise FileNotFoundError(
                f"Pretrained G not found: "
                f"{args.pretrained_g}"
            )

        state_dict = torch.load(args.pretrained_g, map_location=device)
        g_model.load_state_dict(state_dict, strict=True)

        print(
            f"Loaded x{scale} pretrained G from:\n"
            f"{args.pretrained_g}"
        )

    # ===== 2. 构建判别器 (D) =====
    d_model = discriminator_for_vgg(in_channels=3, out_channels=1, channels=64).to(device)

    # ===== 3. 损失函数 =====
    if args.pixel_loss == 'mse':
        pixel_criterion = nn.MSELoss().to(device)

    elif args.pixel_loss == 'l1':
        pixel_criterion = nn.L1Loss().to(device)

    elif args.pixel_loss == 'charbonnier':
        pixel_criterion = CharbonnierLoss().to(device)

    content_criterion = ContentLoss(
        net_cfg_name="vgg19",
        batch_norm=False,
        num_classes=1000,
        model_weights_path="",   # 自动下载 PyTorch 预训练权重
        feature_nodes=["features.35"],  # VGG19 的 conv5_4
        feature_normalize_mean=[0.485, 0.456, 0.406],
        feature_normalize_std=[0.229, 0.224, 0.225]
    ).to(device)
    adversarial_criterion = nn.BCEWithLogitsLoss().to(device)

    # ===== 4. 优化器 & 调度器 =====
    g_optimizer = optim.Adam(g_model.parameters(), lr=args.g_lr, weight_decay=args.weight_decay)
    d_optimizer = optim.Adam(d_model.parameters(), lr=args.d_lr, weight_decay=args.weight_decay)
    g_scheduler = MultiStepLR(g_optimizer, milestones=args.milestones, gamma=args.gamma)
    d_scheduler = MultiStepLR(d_optimizer, milestones=args.milestones, gamma=args.gamma)

    scaler = GradScaler()

    # ===== 5. EMA (可选) =====
    ema_decay = 0.999
    ema_g_model = AveragedModel(
        g_model,
        multi_avg_fn=get_ema_multi_avg_fn(ema_decay),
        use_buffers=True
    )
    # 初始化 EMA 为预训练权重
    ema_g_model.update_parameters(g_model)

    # ===== 6. 评估器 (PSNR/SSIM) =====
    psnr_model = PSNR(crop_border=scale, only_test_y_channel=True).to(device)
    ssim_model = SSIM(crop_border=scale, only_test_y_channel=True).to(device)

    # ===== 7. 日志与保存 =====
    run_tag = f"{args.run_name}_x{scale}"
    raw_save_path = os.path.join(save_dir, f'{run_tag}_best_raw.pth')
    ema_save_path = os.path.join(save_dir, f'{run_tag}_best_ema.pth')
    log_path = os.path.join(save_dir, f'{run_tag}_metrics.csv')
    writer = SummaryWriter(os.path.join("samples", "logs", run_tag))
    config_save_path = os.path.join(save_dir, f'{run_tag}_config.json')

    with open(config_save_path, 'w') as f:
        json.dump(vars(args), f, indent=4)


    # 初始化 CSV
    with open(log_path, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([
            'epoch', 'raw_psnr', 'raw_ssim', 'ema_psnr', 'ema_ssim',
            'g_loss', 'd_loss', 'pixel_loss', 'content_loss', 'adversarial_loss',
            'g_lr', 'd_lr', 'current_adv_weight'
        ])

    # ===== 8. 训练状态 =====
    best_raw_psnr = 0.0
    best_ema_psnr = 0.0

    # ===== 9. 训练循环 =====
    for epoch in range(args.epochs):
        if args.adv_warmup_epochs > 0:
            warmup_factor = min(
                1.0,
                (epoch + 1) / args.adv_warmup_epochs
            )
        else:
            warmup_factor = 1.0

        current_adv_weight = (
            args.adversarial_weight
            * warmup_factor
        )

        g_model.train()
        d_model.train()

        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        g_losses = AverageMeter('G Loss', ':6.6f')
        d_losses = AverageMeter('D Loss', ':6.6f')
        pixel_losses = AverageMeter('Pixel Loss', ':6.6f')
        content_losses = AverageMeter('Content Loss', ':6.6f')
        adv_losses = AverageMeter('Adv Loss', ':6.6f')
        progress = ProgressMeter(len(train_loader), [batch_time, data_time, g_losses, d_losses],
                                 prefix=f"Epoch [{epoch+1}/{args.epochs}]")

        end = time.time()
        for i, (lr_img, hr_img) in enumerate(train_loader):
            lr_img = lr_img.to(device, non_blocking=True)
            hr_img = hr_img.to(device, non_blocking=True)
            data_time.update(time.time() - end)

            # ---- 训练生成器 (G) ----
            # 1) 冻结 D，更新 G
            for p in d_model.parameters():
                p.requires_grad = False

            g_optimizer.zero_grad(set_to_none=True)

            with autocast():
                sr = g_model(lr_img)

                pixel_loss = (
                    args.pixel_weight
                    * pixel_criterion(sr, hr_img)
                )

                content_loss = (
                    args.content_weight
                    * content_criterion(sr, hr_img).mean()
                )

                d_fake_for_g = d_model(sr)

                adversarial_loss = (
                    current_adv_weight
                    * adversarial_criterion(
                        d_fake_for_g,
                        torch.ones_like(d_fake_for_g)
                    )
                )

                g_loss = (
                    pixel_loss
                    + content_loss
                    + adversarial_loss
                )

            scaler.scale(g_loss).backward()
            scaler.step(g_optimizer)

            # ---- 训练判别器 (D) ----
            # 2) 启用 D，使用 sr.detach() 更新 D
            for p in d_model.parameters():
                p.requires_grad = True

            d_optimizer.zero_grad(set_to_none=True)

            with autocast():
                d_real = d_model(hr_img)

                # d_real smooth: Prevent the discriminator from becoming excessively confident.
                real_targets = torch.full_like(
                    d_real,
                    args.real_label
                )

                d_loss_real = adversarial_criterion(
                    d_real,
                    real_targets
                )

                d_fake = d_model(sr.detach())

                d_loss_fake = adversarial_criterion(
                    d_fake,
                    torch.zeros_like(d_fake)
                )

                d_loss = d_loss_real + d_loss_fake

            scaler.scale(d_loss).backward()
            scaler.step(d_optimizer)

            # 两个 optimizer 都 step 后再 update scaler
            scaler.update()

            ema_g_model.update_parameters(g_model)

            # 记录指标
            pixel_losses.update(pixel_loss.item(), lr_img.size(0))
            content_losses.update(content_loss.item(), lr_img.size(0))
            adv_losses.update(adversarial_loss.item(), lr_img.size(0))
            g_losses.update(g_loss.item(), lr_img.size(0))
            d_losses.update(d_loss.item(), lr_img.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0:
                progress.display(i)

        # ---- 验证 ----
        raw_psnr, raw_ssim = evaluate(g_model, valid_loader, psnr_model, ssim_model, device)
        ema_psnr, ema_ssim = evaluate(ema_g_model, valid_loader, psnr_model, ssim_model, device)
        print(f"[Epoch {epoch+1}] RAW: PSNR={raw_psnr:.2f}, SSIM={raw_ssim:.4f} | EMA: PSNR={ema_psnr:.2f}, SSIM={ema_ssim:.4f}")
        print(f"         G Loss={g_losses.avg:.4f} | D Loss={d_losses.avg:.4f}")

        # ---- TensorBoard 日志 ----
        writer.add_scalar('PSNR/Raw', raw_psnr, epoch + 1)
        writer.add_scalar('PSNR/EMA', ema_psnr, epoch + 1)
        writer.add_scalar('Loss/G', g_losses.avg, epoch + 1)
        writer.add_scalar('Loss/D', d_losses.avg, epoch + 1)

        # ---- CSV 日志 ----
        with open(log_path, 'a', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([
                epoch + 1,
                raw_psnr, raw_ssim,
                ema_psnr, ema_ssim,
                g_losses.avg, d_losses.avg,
                pixel_losses.avg,
                content_losses.avg,
                adv_losses.avg,
                g_optimizer.param_groups[0]['lr'],
                d_optimizer.param_groups[0]['lr'],
                current_adv_weight
            ])

        # ---- 保存最佳模型 ----
        raw_improved = raw_psnr > best_raw_psnr
        ema_improved = ema_psnr > best_ema_psnr

        if raw_improved:
            best_raw_psnr = raw_psnr
            torch.save(
                g_model.state_dict(),
                raw_save_path
            )
            print(f"  -> Best RAW model saved (PSNR: {best_raw_psnr:.2f})")

        if ema_improved:
            best_ema_psnr = ema_psnr
            torch.save(
                ema_g_model.module.state_dict(),
                ema_save_path
            )
            print(f"  -> Best EMA model saved (PSNR: {best_ema_psnr:.2f})")

        # 不做早停，仅保存
        # SRGAN 不用 PSNR 做 early stopping: SRGAN 的意义是获得更真实的 perceptual texture
        # SRGAN目标 = perceptual quality

        # 定期保存 checkpoint
        current_epoch = epoch + 1

        should_save_checkpoint = (args.checkpoint_interval > 0 and (
                current_epoch % args.checkpoint_interval == 0
                or current_epoch == args.epochs
                )
            )

        if should_save_checkpoint:
            raw_checkpoint_path = os.path.join(save_dir, f'{run_tag}_epoch_{current_epoch}_raw.pth')
            ema_checkpoint_path = os.path.join(save_dir, f'{run_tag}_epoch_{current_epoch}_ema.pth')
            torch.save(g_model.state_dict(), raw_checkpoint_path)
            torch.save(ema_g_model.module.state_dict(), ema_checkpoint_path)

            print(
                f"  -> Periodic checkpoint saved "
                f"at epoch {current_epoch}"
            )

        g_scheduler.step()
        d_scheduler.step()

    writer.close()
    print("Training finished!")


if __name__ == "__main__":
    main()