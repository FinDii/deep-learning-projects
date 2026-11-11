import os
import pathlib
import itertools
import typing
import random

import numpy as np
import numpy.typing as npt
import torch
import skimage.io
import torch.utils.data

from transform import apply_affine_transform, get_random_affine_matrix_inv


class BrainSegmentationDataset(torch.utils.data.Dataset):
    in_channels = 3
    out_channels = 1

    def __init__(
        self,
        images_dir: str | pathlib.Path,
        is_train: bool = True,
        validation_cases: int = 10,
    ):
        self.is_train = is_train

        images_dir = pathlib.Path(images_dir)

        all_scan_img_paths = [scan_img_path for scan_img_path in images_dir.rglob('*.tif')
            if not scan_img_path.stem.endswith('_mask')]
        all_scan_img_paths = sorted(all_scan_img_paths, key=
            lambda scan_img_path: (scan_img_path.parent.name, int(scan_img_path.stem.rsplit('_', maxsplit=1)[1])))
        assert len(all_scan_img_paths) > 0, f'No images found in {images_dir}'

        all_patient_to_scan_img_paths = {patient: list(scan_img_paths)
            for patient, scan_img_paths in
                itertools.groupby(all_scan_img_paths, key=lambda scan_img_path: scan_img_path.parent.name)}

        self.patients = list(all_patient_to_scan_img_paths.keys())

        # select train/validation subset
        random.seed(42)
        validation_patients = sorted(random.sample(self.patients, k=validation_cases))
        if is_train:
            self.patients = sorted(set(self.patients).difference(validation_patients))
        else:
            self.patients = validation_patients

        # read images
        subset_name = 'train' if is_train else 'validation'
        # print(f'Reading {subset_name} set images ...')
        print(f'Reading {sum(len(all_patient_to_scan_img_paths[patient]) for patient in self.patients)} images from {subset_name} set ...')
        self.patient_scans_float32: list[npt.NDArray[np.float32]] = []
        self.patient_masks_float32: list[npt.NDArray[np.float32]] = []
        for patient in self.patients:
            scans_uint8: list[npt.NDArray[np.uint8]] = []
            masks_uint8: list[npt.NDArray[np.uint8]] = []
            for scan_img_path in all_patient_to_scan_img_paths[patient]:
                scans_uint8.append(skimage.io.imread(scan_img_path))
                masks_uint8.append(skimage.io.imread(scan_img_path.with_stem(scan_img_path.stem + '_mask'), as_gray=True)[..., np.newaxis])

            self.patient_scans_float32.append((np.stack(scans_uint8) / 255).astype(np.float32))   # (n_img, 256, 256, 3)
            self.patient_masks_float32.append((np.stack(masks_uint8) / 255).astype(np.float32))   # (n_img, 256, 256, 1)

        # probabilities for sampling slices based on masks
        self.slice_weights = [np.sum(masks_float32, axis=(-1, -2, -3)) for masks_float32 in self.patient_masks_float32]
        self.slice_weights = [(s + (s.sum() * 0.1 / len(s))) / (s.sum() * 1.1) for s in self.slice_weights]

        # create global index for patient and slice (idx -> (p_idx, s_idx))
        self.patient_n_imgs = [scans_float32.shape[0] for scans_float32 in self.patient_scans_float32]
        self.patient_slice_index = np.stack([
            np.repeat(np.arange(len(self.patient_n_imgs)), self.patient_n_imgs),
            np.concatenate([np.arange(n_imgs) for n_imgs in self.patient_n_imgs]),
        ], axis=-1)

    def __len__(self):
        return len(self.patient_slice_index)

    def __getitem__(self, idx: int):
        if self.is_train:
            i_patient = np.random.randint(len(self.patients))
            slice_n = np.random.choice(
                range(self.patient_n_imgs[i_patient]),
                p=self.slice_weights[i_patient])
        else:
            i_patient, slice_n = self.patient_slice_index[idx]

        scan_float32 = self.patient_scans_float32[i_patient][slice_n]
        mask_float32 = self.patient_masks_float32[i_patient][slice_n]

        # fix dimensions (C, H, W)
        scan_float32 = scan_float32.transpose(2, 0, 1)
        mask_float32 = mask_float32.transpose(2, 0, 1)

        scan = torch.from_numpy(scan_float32.astype(np.float32))
        mask = torch.from_numpy(mask_float32.astype(np.float32))

        if self.is_train:
            g_inv = get_random_affine_matrix_inv(  # random affine transformation matrix
                reflection     =True,
                angle_range_deg=(-45.  , 45.  ),
                scale_range    =(  0.75,  1.33),
                translate_range=( -0.2 ,  0.2 ))
        else:
            g_inv = torch.eye(3, dtype=torch.float32)

        # return tensors
        return scan, mask, g_inv


