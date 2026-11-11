
import os
import sys
sys.path.append('/content/SRGAN-PyTorch')

import argparse
import torch
from torch.utils.data import DataLoader
from sr_dataset_exp_scale import SRDatasetConfig, SRDataset
from model_srgan import SRResNet
from image_quality_assessment import PSNR, SSIM
from utils import AverageMeter

def get_hr_folder(base_hr, scale):
    """
    如果 base_hr/HR_cropped/X{scale} 存在，则使用裁剪后的 HR，
    否则回退到原始 base_hr。
    """
    hr_cropped_dir = os.path.join(base_hr, 'HR_cropped', f'X{scale}')
    if os.path.isdir(hr_cropped_dir):
        return hr_cropped_dir
    return base_hr

def test_on_benchmark(model_path, benchmark_name, config_params, scale=4):
    """
    测试单个基准数据集
    Args:
        scale (int): 上采样倍数，2,3,4
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型
    model = SRResNet(channels=64, num_rcb=16, upscale=scale).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 数据集配置
    test_config = SRDatasetConfig(
        csv_folder=config_params['csv_folder'],
        HR_data_folder=config_params['image_folder'],
        LR_data_folder=config_params['lr_folder'],
        crop_size=-1,
        scaling_factor=scale,
        patch_num=1,
        cache=False,
        scale=scale,
        color_augment=False,
        noise_augment=False
    )
    test_dataset = SRDataset(benchmark_name, test_config)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    # 评估器
    psnr_model = PSNR(crop_border=scale, only_test_y_channel=True).to(device)
    ssim_model = SSIM(crop_border=scale, only_test_y_channel=True).to(device)

    psnr_meter = AverageMeter('PSNR', ':6.2f')
    ssim_meter = AverageMeter('SSIM', ':6.4f')

    with torch.no_grad():
        for lr_img, hr_img in test_loader:
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)
            sr = model(lr_img)

            # 如果尺寸不一致（仅发生在未裁剪对齐的 benchmark 图像时），取最小尺寸对齐
            if sr.shape != hr_img.shape:
                min_h = min(sr.shape[2], hr_img.shape[2])
                min_w = min(sr.shape[3], hr_img.shape[3])
                sr = sr[:, :, :min_h, :min_w]
                hr_img = hr_img[:, :, :min_h, :min_w]

            psnr = psnr_model(sr, hr_img)
            ssim = ssim_model(sr, hr_img)
            psnr_meter.update(psnr.item())
            ssim_meter.update(ssim.item())

    print(f"{benchmark_name}: PSNR = {psnr_meter.avg:.2f} dB, SSIM = {ssim_meter.avg:.4f}")
    return psnr_meter.avg, ssim_meter.avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test SRResNet on benchmarks")
    parser.add_argument('--model_path', type=str,
                        default="/content/drive/MyDrive/DIV2K_Results/srresnet_x4_best_raw.pth",
                        help="Path to the model checkpoint (.pth)")
    parser.add_argument('--scale', type=int, default=4, choices=[2,3,4],
                        help="Upscaling factor of the model")
    parser.add_argument('--csv_folder', type=str,
                        default="/content/drive/MyDrive/DIV2K_Results/output_csv",
                        help="Folder containing CSV files")
    args = parser.parse_args()

    scale = args.scale

    # 动态构造基准数据集的路径（替换 X4 为 X{scale}）
    base_hr_root = "/content/drive/MyDrive/datasets/DIV2K_HRLR/benchmark"
    base_lr_root = "/content/drive/MyDrive/datasets/DIV2K_HRLR/benchmark"

    benchmarks = {
        "Set5": {
            "image_folder": get_hr_folder(f"{base_hr_root}/Set5/HR", scale),
            "lr_folder": f"{base_lr_root}/Set5/LR_bicubic/X{scale}",
            "csv_folder": args.csv_folder
        },
        "Set14": {
            "image_folder": get_hr_folder(f"{base_hr_root}/Set14/HR", scale),
            "lr_folder": f"{base_lr_root}/Set14/LR_bicubic/X{scale}",
            "csv_folder": args.csv_folder
        },
        "Urban100": {
            "image_folder": get_hr_folder(f"{base_hr_root}/Urban100/HR", scale),
            "lr_folder": f"{base_lr_root}/Urban100/LR_bicubic/X{scale}",
            "csv_folder": args.csv_folder
        },
        "BSDS100": {
            "image_folder": get_hr_folder(f"{base_hr_root}/B100/HR", scale),
            "lr_folder": f"{base_lr_root}/B100/LR_bicubic/X{scale}",
            "csv_folder": args.csv_folder
        }
    }

    for name, paths in benchmarks.items():
        test_on_benchmark(
            model_path=args.model_path,
            benchmark_name=name,
            config_params=paths,
            scale=scale
        )