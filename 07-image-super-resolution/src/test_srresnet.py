
import os
import sys
sys.path.append('/content/SRGAN-PyTorch')

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sr_dataset import SRDatasetConfig, SRDataset
from model import srresnet_x4
from image_quality_assessment import PSNR, SSIM
from utils import AverageMeter

def test_on_benchmark(model_path, benchmark_name, config_params):
    """
    测试单个基准数据集
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = srresnet_x4().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_config = SRDatasetConfig(
        csv_folder="/content/drive/MyDrive/DIV2K_Results/output_csv",
        HR_data_folder=config_params['image_folder'],
        LR_data_folder=config_params['lr_folder'],
        crop_size=-1,
        scaling_factor=4,
        patch_num=1,
        cache=False
    )
    test_dataset = SRDataset(benchmark_name, test_config)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    psnr_model = PSNR(crop_border=4, only_test_y_channel=True).to(device)
    ssim_model = SSIM(crop_border=4, only_test_y_channel=True).to(device)

    psnr_meter = AverageMeter('PSNR', ':6.2f')
    ssim_meter = AverageMeter('SSIM', ':6.4f')

    with torch.no_grad():
        for lr_img, hr_img in test_loader:
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)
            sr = model(lr_img)
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
    args = parser.parse_args()

    model_path = args.model_path

    benchmarks = {
        "Set5": {
            "image_folder": "/content/drive/MyDrive/datasets/DIV2K_HRLR/benchmark/Set5/HR",
            "lr_folder": "/content/drive/MyDrive/datasets/DIV2K_HRLR/benchmark/Set5/LR_bicubic/X4"
        },
        "Set14": {
            "image_folder": "/content/drive/MyDrive/datasets/DIV2K_HRLR/benchmark/Set14/HR",
            "lr_folder": "/content/drive/MyDrive/datasets/DIV2K_HRLR/benchmark/Set14/LR_bicubic/X4"
        },
        "Urban100": {
            "image_folder": "/content/drive/MyDrive/datasets/DIV2K_HRLR/benchmark/Urban100/HR",
            "lr_folder": "/content/drive/MyDrive/datasets/DIV2K_HRLR/benchmark/Urban100/LR_bicubic/X4"
        },
        "BSDS100": {
            "image_folder": "/content/drive/MyDrive/datasets/DIV2K_HRLR/benchmark/B100/HR",
            "lr_folder": "/content/drive/MyDrive/datasets/DIV2K_HRLR/benchmark/B100/LR_bicubic/X4"
        }
    }

    for name, paths in benchmarks.items():
        test_on_benchmark(model_path, name, paths)