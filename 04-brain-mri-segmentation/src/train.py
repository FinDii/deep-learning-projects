'''
新建python 3.13环境（以conda为例）
    conda create -n hw4 python=3.13 -y
    conda activate hw4

安装torch，注意cuda版本适配
    pip install torch==2.9.* torchvision --index-url https://download.pytorch.org/whl/cu128

安装其他依赖库
    pip install ipykernel==7.* matplotlib==3.* scipy==1.17.* numpy==2.3.* scikit-image==0.26.* tensorboard==2.20.* tqdm==4.* -i https://pypi.tuna.tsinghua.edu.cn/simple
'''

import os
import pathlib
import time
import json
import tqdm
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from dataset import BrainSegmentationDataset
from transform import apply_affine_transform, get_random_affine_matrix_inv
from loss import DiceLoss
from unet import UNet
from utils import log_images, dsc, dataloader_worker_init


#    █  ▄▄▄▄█▄     ▄▀  ▄    ▀▄ █▄▀  █    ▄▄▄▄▄█▄▄▄▄▄▄  █▀▀█▀▀█▀█
#  ▀▀█▀▀ █  █    ▀▀▀█▀▀▀▀▄  ▀▀██▀▀ █▀▀█▀  ▄█▄█▄     █  ▀▀▀▀█▀▀▀▀
# ▀▀▀█▀▄▀ ▀▀   ▀▀▀█▀▀▀█▀▀▀▀ ▄▀ █▀ ▀  ▄▀   █ █ █ ▄▄▄▄█ ▀▀▀▀▀█▀▀▀▀▀▀
#  █ █▄ █▀▀▀█   ▄▀▄▄▀ ▄▀▀▄▄ ▄▄█▄▄▄ █ █    █▀ ▀█ █       █▀▀█▀▀█
# ▄▀▄█  █▄▄▄█  ▀ ▀▄▄▀▀ ▄▄    ▀▄▄▀   █     █▀▀▀█ █   ▄   █▄▄▄█▄█
# ▀  ▀▀▄▄▄▄▄▄▄  ▄▄▄▄▄▀▀     ▄▄▀ ▀▄▄▀ ▀▄▄  █▀▀▀█ ▀▄▄▄█ ▄▄█▄▄▄▄▄█▄█▄

ckpt_root_dir = './ckpt'
# 跑基线（原始 MaxPool）
ckpt_name     = 'baseline_maxpool'

batch_size    = 16
epochs        = 100
lr            = 1e-4
vis_freq      = 10      # 两次可视化预测结果的间隔
device        = torch.device('cuda:0')
workers       = 2

def find_data_root():
    # 如果在 Kaggle 环境中
    if os.path.exists('/kaggle/input'):
        for root, dirs, files in os.walk('/kaggle/input'):
            # Note on Dataset, File, Check, Data. Refer to the archived source for the detailed original comment.
            if any(d.startswith('TCGA') and os.path.isdir(os.path.join(root, d)) for d in dirs):
                return root
            # Note on Dataset, File, Check, Data. Refer to the archived source for the detailed original comment.
            if any(f.endswith('.tif') for f in files):
                # Note on Dataset, Data. Refer to the archived source for the detailed original comment.
                return root
        raise RuntimeError("No brain MRI dataset found under /kaggle/input")
    else:
        # 本地环境
        return './kaggle_3m'

data_root_dir = find_data_root()
print(f"Using data root: {data_root_dir}")

# --- 自动绘图 ---
def plot_curves(ckpt_dir, json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    loss_per_epoch = data['loss_per_epoch_per_patient']  # list of lists: [epoch][patient]
    dsc_per_epoch  = data['dsc_per_epoch_per_patient']

    epochs = len(loss_per_epoch)
    n_patients = len(loss_per_epoch[0]) if epochs > 0 else 0

    # Note on Compute. Refer to the archived source for the detailed original comment.
    mean_loss = [np.mean(epoch_losses) for epoch_losses in loss_per_epoch]
    std_loss  = [np.std(epoch_losses)  for epoch_losses in loss_per_epoch]
    mean_dsc  = [np.mean(epoch_dscs)   for epoch_dscs  in dsc_per_epoch]
    std_dsc   = [np.std(epoch_dscs)    for epoch_dscs  in dsc_per_epoch]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss 曲线（带标准差带）
    ax1.plot(range(epochs), mean_loss, 'r-', label='Mean Validation Loss')
    ax1.fill_between(range(epochs), 
                     np.array(mean_loss) - np.array(std_loss),
                     np.array(mean_loss) + np.array(std_loss),
                     color='r', alpha=0.2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # DSC 曲线（带标准差带）
    ax2.plot(range(epochs), mean_dsc, 'b-', label='Mean Validation DSC')
    ax2.fill_between(range(epochs),
                     np.array(mean_dsc) - np.array(std_dsc),
                     np.array(mean_dsc) + np.array(std_dsc),
                     color='b', alpha=0.2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('DSC')
    ax2.set_title('Validation DSC')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plot_path = ckpt_dir / 'loss_dsc_curves.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f'Saved curves to {plot_path}')


#  ▀▄  █  ▄  █   █ ▄▄▄█▄▄▄▄   █ ▄▄█▄▄█▄▄ ▄▄▄▄▄▄▄ ▄ █
#      █  █  █  █ ▄  █ ▄    ▄▄█▄ ▄█▄▄█▄  ▄▄█▄█▄▄ █ █
# ▀▀█  █  █  █ ▀▀█▀ █▀▀█▀    ▄█▄ █▄▄▄▄█    █ █   █ █
#   █ ▄█  █  █ ▄█▄▄▀▀▀▀█▀▀▀ █ █ ▀█▄▄▄▄█  ▄▀  █ ▄  ▄█
#   █▀ █  █  █   ▄▄ ▄▀ █ ▀▄   █ ▄▄▄█▄▄▄▄   ▄▄▄▄█▄▄▄▄
#    ▄▀      █ ▀▀ ▄▀ ▀▄▀  ▀   █ ▄▄▀  ▀▄▄ ▄▄▄▄▄▄█▄▄▄▄▄

if __name__ == '__main__':
    timestamp_str = time.strftime(r'%y%m%d-%H%M%S', time.localtime())
    ckpt_dir = pathlib.Path(ckpt_root_dir, f'{timestamp_str} {ckpt_name}')
    print(f'Save directory: {ckpt_dir}')
    os.makedirs(ckpt_dir, exist_ok=True)

    # Note on Load, Data. Refer to the archived source for the detailed original comment.
    dataset_train = BrainSegmentationDataset(data_root_dir, is_train=True)
    dataset_valid = BrainSegmentationDataset(data_root_dir, is_train=False)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True,  persistent_workers=True, pin_memory=True, num_workers=workers, worker_init_fn=dataloader_worker_init,)
    loader_valid = DataLoader(dataset_valid, batch_size=batch_size,               drop_last=False, persistent_workers=True, pin_memory=True, num_workers=workers, worker_init_fn=dataloader_worker_init,)

    # Note on Model, Create. Refer to the archived source for the detailed original comment.
    unet = UNet(in_channels=BrainSegmentationDataset.in_channels, out_channels=BrainSegmentationDataset.out_channels).to(device)

    dsc_loss = DiceLoss()
    optimizer = optim.Adam(unet.parameters(), lr=lr)

    summarywriter = SummaryWriter(pathlib.Path(ckpt_dir, 'logs'))

    # Note on Training. Refer to the archived source for the detailed original comment.
    loss_per_epoch_per_patient: list[list[float]] = []
    dsc_per_epoch_per_patient:  list[list[float]] = []

    best_validation_dsc = 0.
    for epoch in range(epochs):
        # train
        _ = unet.train()

        losses: list[float] = []
        epoch_loader_train = tqdm.tqdm(loader_train, desc=f'E#{epoch}', postfix='init')
        for i, (xs, ys_true, gs_inv) in enumerate(epoch_loader_train):
            xs      = xs     .to(device)
            ys_true = ys_true.to(device)
            gs_inv  = gs_inv .to(device)  # random affine transformation matrix

            with torch.no_grad():
                # data augmentation with random affine transformation
                xs      = apply_affine_transform(xs,      gs_inv, interpolation='bilinear', edge_fill_value=0.)
                ys_true = apply_affine_transform(ys_true, gs_inv, interpolation='nearest',  edge_fill_value=0.)

            ys_pred: torch.Tensor = unet(xs)
            loss: torch.Tensor = dsc_loss(ys_pred, ys_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            epoch_loader_train.set_postfix_str(f'L={np.mean(losses):0.4f}')

        mean_loss = np.mean(losses)
        summarywriter.add_scalar('train loss', mean_loss, epoch)

        # validation
        _ = unet.eval()

        n_slices_per_patient = np.bincount(dataset_valid.patient_slice_index[..., 0])
        i0_slice_per_patient = np.cumsum(n_slices_per_patient) - n_slices_per_patient
        
        loss_per_patient: list[float] = []
        dsc_per_patient:  list[float] = []
        for i_patient, (i0_slice, n_slices) in tqdm.tqdm(enumerate(zip(i0_slice_per_patient, n_slices_per_patient)), desc=f'  Validating'):
            patient_name: str = dataset_valid.patients[i_patient]

            patient_losses:  list[float] = []
            patient_ys_pred: list[np.ndarray] = []
            patient_ys_true: list[np.ndarray] = []
            for i_slice in range(i0_slice, i0_slice + n_slices):
                x, y_true, g_inv = dataset_valid[i_slice]
                xs      = x     [None].to(device)
                ys_true = y_true[None].to(device)

                with torch.no_grad():
                    ys_pred: torch.Tensor = unet(xs)
                    loss = dsc_loss(ys_pred, ys_true)

                    patient_losses .append(loss.item())
                    patient_ys_pred.append(ys_pred.squeeze(0).detach().cpu().numpy())
                    patient_ys_true.append(ys_true.squeeze(0).detach().cpu().numpy())

                    if (epoch % vis_freq == 0) or (epoch == epochs - 1):
                        img, = log_images(xs, ys_true, ys_pred)
                        summarywriter.add_image(f'val images/{i_patient} {patient_name}/{i_slice}', img, epoch, dataformats='HWC')

            loss_per_patient.append(float(np.mean(patient_losses)))
            dsc_per_patient .append(float(dsc(np.stack(patient_ys_pred), np.stack(patient_ys_true))))

        loss_per_epoch_per_patient.append(loss_per_patient)
        dsc_per_epoch_per_patient .append(dsc_per_patient)

        mean_loss = np.mean(loss_per_patient)
        mean_dsc  = np.mean(dsc_per_patient)
        summarywriter.add_scalar('val loss', mean_loss, epoch)
        summarywriter.add_scalar('val dsc',  mean_dsc,  epoch)
        print(f'  val loss: {mean_loss:0.4f}, val dsc: {mean_dsc:0.4f}')

        # save best model
        if mean_dsc > best_validation_dsc:
            best_validation_dsc = float(mean_dsc)
            torch.save(unet.state_dict(), pathlib.Path(ckpt_dir, 'unet.pt'))

    print(f'Best validation mean DSC: {best_validation_dsc:4f}')

    with open(pathlib.Path(ckpt_dir, 'losses_dsc_values.json'), 'wt', encoding='utf-8') as f:
        json.dump({
            'best_dsc': best_validation_dsc,
            'patient_names': dataset_valid.patients,
            'loss_per_epoch_per_patient': loss_per_epoch_per_patient,
            'dsc_per_epoch_per_patient':  dsc_per_epoch_per_patient,
        }, f, indent=2, ensure_ascii=False)


    # 调用绘图函数
    plot_curves(ckpt_dir, ckpt_dir / 'losses_dsc_values.json')
