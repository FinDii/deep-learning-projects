
'''
参数化训练脚本 - 支持多种配置对比实验
'''
import os
import pathlib
import time
import json
import tqdm
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

import sys
sys.path.append('/kaggle/input/notebooks/fdiiiiii/mri-skimage') 
sys.path.append('/kaggle/input/notebooks/fdiiiiii/mri-skimage-compare2') 

from dataset_exp import BrainSegmentationDataset
from transform import apply_affine_transform, get_random_affine_matrix_inv
from utils import log_images, dsc, dataloader_worker_init
from unet_exp import UNet, UNetWithAttention, ResUNet
from loss_exp import DiceLoss, FocalLoss, DiceBCELoss
from transform_exp import ElasticTransform
from transform import apply_affine_transform, get_random_affine_matrix_inv


def safe_largest_connected_component(img):
    from scipy.ndimage import label
    labeled_array, num_features = label(img, None)
    if num_features == 0:
        return np.zeros_like(img, dtype=bool)
    component_sizes = np.bincount(labeled_array.ravel())[1:]
    return labeled_array == (np.argmax(component_sizes) + 1)

# 覆盖 utils 中的函数
import utils
utils.largest_connected_component = safe_largest_connected_component

def parse_args():
    parser = argparse.ArgumentParser(description='Brain MRI Segmentation Training')
    
    # 数据相关
    parser.add_argument('--data_root', type=str, default=None,
                        help='Data root directory (auto-detect if None)')
    parser.add_argument('--elastic', action='store_true',
                    help='Enable elastic deformation augmentation')
    
    # 网络结构
    parser.add_argument('--network', type=str, default='unet',
                        choices=['unet', 'attention_unet', 'resunet'],
                        help='Network architecture')
    parser.add_argument('--downsample', type=str, default='maxpool',
                        choices=['maxpool', 'stride_conv'],
                        help='Downsampling mode')
    parser.add_argument('--upsample', type=str, default='convtranspose',
                        choices=['convtranspose', 'bilinear', 'pixelshuffle'],
                        help='Upsampling mode')
    
    # 损失函数
    parser.add_argument('--loss', type=str, default='dice',
                        choices=['dice', 'focal', 'dice_bce'],
                        help='Loss function')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal loss gamma parameter')

    # 损失函数权重（仅用于 dice_bce）
    parser.add_argument('--dice_weight', type=float, default=1.0,
                        help='Dice loss weight for DiceBCELoss')
    parser.add_argument('--bce_weight', type=float, default=0.5,
                        help='BCE loss weight for DiceBCELoss')
    
    # 优化器
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'adamw', 'sgd'],
                        help='Optimizer')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    
    # 学习率调度
    parser.add_argument('--scheduler', type=str, default='none',
                        choices=['none', 'step', 'plateau', 'cosine'],
                        help='Learning rate scheduler')
    parser.add_argument('--step_size', type=int, default=30,
                        help='StepLR step size')
    parser.add_argument('--plateau_patience', type=int, default=10,
                        help='ReduceLROnPlateau patience')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--vis_freq', type=int, default=10,
                        help='Visualization frequency')
    parser.add_argument('--workers', type=int, default=2,
                        help='DataLoader workers')
    
    # 混合精度
    parser.add_argument('--amp', action='store_true',
                        help='Enable mixed precision training')
    
    # 实验标识
    parser.add_argument('--ckpt_name', type=str, default=None,
                        help='Custom checkpoint name')
    parser.add_argument('--comment', type=str, default='',
                        help='Additional comment for experiment name')
    
    return parser.parse_args()


def get_loss_function(args):
    """根据参数返回损失函数"""
    if args.loss == 'dice':
        return DiceLoss()
    elif args.loss == 'focal':
        return FocalLoss(gamma=args.focal_gamma)
    elif args.loss == 'dice_bce':
        return DiceBCELoss(dice_weight=args.dice_weight, bce_weight=args.bce_weight)
    else:
        raise ValueError(f"Unknown loss: {args.loss}")


def get_optimizer(model, args):
    """根据参数返回优化器"""
    if args.optimizer == 'adam':
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        return optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                        weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")


def get_scheduler(optimizer, args):
    """根据参数返回学习率调度器"""
    if args.scheduler == 'none':
        return None
    elif args.scheduler == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    elif args.scheduler == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                     patience=args.plateau_patience, factor=0.5)
    elif args.scheduler == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler}")


def get_network(args):
    """根据参数返回网络模型"""
    in_channels = BrainSegmentationDataset.in_channels
    out_channels = BrainSegmentationDataset.out_channels
    
    if args.network == 'unet':
        return UNet(in_channels=in_channels, out_channels=out_channels,
                   downsample_mode=args.downsample, upsample_mode=args.upsample)
    elif args.network == 'attention_unet':
        return UNetWithAttention(in_channels=in_channels, out_channels=out_channels,
                                downsample_mode=args.downsample, upsample_mode=args.upsample)
    elif args.network == 'resunet':
        return ResUNet(in_channels=in_channels, out_channels=out_channels,
                      downsample_mode=args.downsample, upsample_mode=args.upsample)
    else:
        raise ValueError(f"Unknown network: {args.network}")


def find_data_root():
    """自动检测数据根目录"""
    if os.path.exists('/kaggle/input'):
        for root, dirs, files in os.walk('/kaggle/input'):
            if any(d.startswith('TCGA') and os.path.isdir(os.path.join(root, d)) for d in dirs):
                return root
            if any(f.endswith('.tif') for f in files):
                return root
        raise RuntimeError("No brain MRI dataset found under /kaggle/input")
    else:
        return './kaggle_3m'


def plot_curves(ckpt_dir, json_path, config_name):
    """绘制训练曲线"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    train_losses = data.get('train_loss_per_epoch', [])
    loss_per_epoch = data['loss_per_epoch_per_patient']
    dsc_per_epoch = data['dsc_per_epoch_per_patient']

    epochs = len(loss_per_epoch)
    
    mean_val_loss = [np.mean(epoch_losses) for epoch_losses in loss_per_epoch]
    std_val_loss = [np.std(epoch_losses) for epoch_losses in loss_per_epoch]
    mean_dsc = [np.mean(epoch_dscs) for epoch_dscs in dsc_per_epoch]
    std_dsc = [np.std(epoch_dscs) for epoch_dscs in dsc_per_epoch]

    if len(train_losses) != epochs:
        if len(train_losses) > epochs:
            train_losses = train_losses[:epochs]
        else:
            train_losses += [np.nan] * (epochs - len(train_losses))

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='black')
    ax1.plot(range(epochs), train_losses, color='orange', linestyle='-', marker='.', label='Train Loss')
    ax1.plot(range(epochs), mean_val_loss, color='red', linestyle='-', marker='.', label='Val Loss')
    ax1.fill_between(range(epochs),
                     np.array(mean_val_loss) - np.array(std_val_loss),
                     np.array(mean_val_loss) + np.array(std_val_loss),
                     color='red', alpha=0.2)
    ax1.tick_params(axis='y')
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('DSC', color=color)
    ax2.plot(range(epochs), mean_dsc, color=color, linestyle='-', marker='.', label='Val DSC')
    ax2.fill_between(range(epochs),
                     np.array(mean_dsc) - np.array(std_dsc),
                     np.array(mean_dsc) + np.array(std_dsc),
                     color=color, alpha=0.2)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    best_dsc = max(mean_dsc)
    best_epoch = np.argmax(mean_dsc)
    ax2.plot(best_epoch, best_dsc, 'g*', markersize=15, label=f'Best DSC={best_dsc:.4f}')
    ax2.legend(loc='upper right')

    plt.title(f'Training Curves - {config_name}')
    fig.tight_layout()

    plot_path = ckpt_dir / 'loss_dsc_curves.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved curves to {plot_path}')


if __name__ == '__main__':
    args = parse_args()
    
    # 生成配置名称
    config_parts = [args.network, args.downsample, args.upsample, args.loss, args.optimizer]
    if args.scheduler != 'none':
        config_parts.append(args.scheduler)
    if args.amp:
        config_parts.append('amp')
    config_name = '_'.join(config_parts)
    
    if args.ckpt_name is None:
        ckpt_name = config_name + args.comment
    else:
        ckpt_name = args.ckpt_name

    # 设置设备
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    print(f"Configuration: {config_name}")

    # 数据路径
    if args.data_root is None:
        data_root_dir = find_data_root()
    else:
        data_root_dir = args.data_root
    print(f"Data root: {data_root_dir}")

    # 创建检查点目录
    timestamp_str = time.strftime(r'%y%m%d-%H%M%S', time.localtime())
    ckpt_dir = pathlib.Path('./ckpt', f'{timestamp_str} {ckpt_name}')
    print(f'Save directory: {ckpt_dir}')
    os.makedirs(ckpt_dir, exist_ok=True)

    # 保存配置到 JSON（便于复现）
    with open(ckpt_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # 加载数据
    dataset_train = BrainSegmentationDataset(
        data_root_dir, is_train=True,
        use_elastic=args.elastic   # 根据参数启用
    )
    dataset_valid = BrainSegmentationDataset(data_root_dir, is_train=False)
    loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, 
                              drop_last=True, persistent_workers=True, 
                              pin_memory=True, num_workers=args.workers,
                              worker_init_fn=dataloader_worker_init)
    loader_valid = DataLoader(dataset_valid, batch_size=args.batch_size,
                              drop_last=False, persistent_workers=True,
                              pin_memory=True, num_workers=args.workers,
                              worker_init_fn=dataloader_worker_init)

    # 创建模型
    unet = get_network(args).to(device)
    print(f"Model parameters: {sum(p.numel() for p in unet.parameters()):,}")

    # 损失函数
    criterion = get_loss_function(args)
    
    # 优化器
    optimizer = get_optimizer(unet, args)
    
    # 学习率调度器
    scheduler = get_scheduler(optimizer, args)
    
    # 混合精度训练（修复 Focal+AMP 冲突）
    if args.loss == 'focal' and args.amp:
        print("Warning: Focal Loss is not compatible with AMP. Disabling AMP for this run.")
        args.amp = False
        scaler = None
    else:
        scaler = torch.cuda.amp.GradScaler() if args.amp and torch.cuda.is_available() else None
    if args.amp and not torch.cuda.is_available():
        print("Warning: AMP requires CUDA, disabling.")

    summarywriter = SummaryWriter(pathlib.Path(ckpt_dir, 'logs'))

    # 训练循环
    loss_per_epoch_per_patient = []
    dsc_per_epoch_per_patient = []
    train_losses_per_epoch = []

    best_validation_dsc = 0.0

    for epoch in range(args.epochs):
        unet.train()
        losses = []
        epoch_loader_train = tqdm.tqdm(loader_train, desc=f'E#{epoch}', postfix='init')
        
        for i, (xs, ys_true, gs_inv) in enumerate(epoch_loader_train):
            xs = xs.to(device)
            ys_true = ys_true.to(device)
            gs_inv = gs_inv.to(device)

            with torch.no_grad():
                xs = apply_affine_transform(xs, gs_inv, interpolation='bilinear', edge_fill_value=0.)
                ys_true = apply_affine_transform(ys_true, gs_inv, interpolation='nearest', edge_fill_value=0.)

            optimizer.zero_grad()
            
            # 混合精度前向传播
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    ys_pred = unet(xs)
                    loss = criterion(ys_pred, ys_true)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                ys_pred = unet(xs)
                loss = criterion(ys_pred, ys_true)
                loss.backward()
                optimizer.step()

            losses.append(loss.item())
            epoch_loader_train.set_postfix_str(f'L={np.mean(losses):.4f}')

        mean_train_loss = np.mean(losses)
        train_losses_per_epoch.append(mean_train_loss)
        summarywriter.add_scalar('train/loss', mean_train_loss, epoch)

        # 验证阶段
        unet.eval()
        
        n_slices_per_patient = np.bincount(dataset_valid.patient_slice_index[..., 0])
        i0_slice_per_patient = np.cumsum(n_slices_per_patient) - n_slices_per_patient
        
        loss_per_patient = []
        dsc_per_patient = []
        
        for i_patient, (i0_slice, n_slices) in enumerate(zip(i0_slice_per_patient, n_slices_per_patient)):
            patient_name = dataset_valid.patients[i_patient]
            
            patient_losses = []
            patient_ys_pred = []
            patient_ys_true = []
            
            for i_slice in range(i0_slice, i0_slice + n_slices):
                x, y_true, g_inv = dataset_valid[i_slice]
                xs = x[None].to(device)
                ys_true = y_true[None].to(device)

                with torch.no_grad():
                    if scaler is not None:
                        with torch.cuda.amp.autocast():
                            ys_pred = unet(xs)
                            loss = criterion(ys_pred, ys_true)
                    else:
                        ys_pred = unet(xs)
                        loss = criterion(ys_pred, ys_true)

                    patient_losses.append(loss.item())
                    patient_ys_pred.append(ys_pred.squeeze(0).detach().cpu().numpy())
                    patient_ys_true.append(ys_true.squeeze(0).detach().cpu().numpy())

                    if (epoch % args.vis_freq == 0) or (epoch == args.epochs - 1):
                        img, = log_images(xs, ys_true, ys_pred)
                        summarywriter.add_image(f'val/images/{i_patient}_{patient_name}/{i_slice}', 
                                               img, epoch, dataformats='HWC')

            loss_per_patient.append(float(np.mean(patient_losses)))
            dsc_per_patient.append(float(dsc(np.stack(patient_ys_pred), np.stack(patient_ys_true))))

        loss_per_epoch_per_patient.append(loss_per_patient)
        dsc_per_epoch_per_patient.append(dsc_per_patient)

        mean_val_loss = np.mean(loss_per_patient)
        mean_dsc = np.mean(dsc_per_patient)
        
        summarywriter.add_scalar('val/loss', mean_val_loss, epoch)
        summarywriter.add_scalar('val/dsc', mean_dsc, epoch)
        summarywriter.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f'  val loss: {mean_val_loss:.4f}, val dsc: {mean_dsc:.4f}')

        # 学习率调度
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(mean_val_loss)
            else:
                scheduler.step()

        # 保存最佳模型
        if mean_dsc > best_validation_dsc:
            best_validation_dsc = mean_dsc
            torch.save(unet.state_dict(), ckpt_dir / 'unet.pt')
            print(f'  New best model saved (DSC: {best_validation_dsc:.4f})')

    print(f'Best validation mean DSC: {best_validation_dsc:.6f}')

    # 保存训练历史
    with open(ckpt_dir / 'losses_dsc_values.json', 'wt', encoding='utf-8') as f:
        json.dump({
            'best_dsc': best_validation_dsc,
            'patient_names': dataset_valid.patients,
            'train_loss_per_epoch': train_losses_per_epoch,
            'loss_per_epoch_per_patient': loss_per_epoch_per_patient,
            'dsc_per_epoch_per_patient': dsc_per_epoch_per_patient,
        }, f, indent=2, ensure_ascii=False)

    # 绘制曲线
    plot_curves(ckpt_dir, ckpt_dir / 'losses_dsc_values.json', config_name)
    summarywriter.close()
