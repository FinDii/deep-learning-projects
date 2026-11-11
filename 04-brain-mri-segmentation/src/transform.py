import typing
import numpy as np
import torch
from skimage.transform import rescale, rotate
from torchvision.transforms import Compose


def make_affine_matrix_inv(reflected: bool, angle_deg: float, scale_ratio: float, i_offset: float, j_offset: float) -> np.ndarray:
    angle_rad = angle_deg * (np.pi / 180)
    g_inv = np.eye(3)

    # x-reflection matrix
    if reflected:
        g_inv[1, 1] = -1

    # inverse rotation matrix
    g_inv = g_inv @ np.array([
        [np.cos(angle_rad),  np.sin(angle_rad), 0],
        [-np.sin(angle_rad), np.cos(angle_rad), 0],
        [0,                  0,                 1]])

    # inverse scale matrix
    g_inv = g_inv @ np.array([
        [1 / scale_ratio, 0,               0],
        [0,               1 / scale_ratio, 0],
        [0,               0,               1]])

    # inverse translation matrix
    g_inv = g_inv @ np.array([
        [1, 0, -i_offset],
        [0, 1, -j_offset],
        [0, 0, 1]])

    return g_inv

def _lerp(alpha, x, y):
    return x + alpha * (y - x)

def get_random_affine_matrix_inv(reflection: bool, angle_range_deg: tuple[float, float], scale_range: tuple[float, float], translate_range: tuple[float, float]) -> torch.Tensor:
    reflected   = np.random.rand() >= 0.5 if reflection else False
    angle_deg   = _lerp(np.random.rand(), *angle_range_deg)
    scale_ratio = 2 ** _lerp(np.random.rand(), *np.log2(scale_range))
    i_offset    = _lerp(np.random.rand(), *translate_range)
    j_offset    = _lerp(np.random.rand(), *translate_range)
    g_inv = make_affine_matrix_inv(reflected, angle_deg, scale_ratio, i_offset, j_offset)
    return torch.from_numpy(g_inv.astype(np.float32))


def apply_affine_transform(images: torch.Tensor, g_inv: torch.Tensor, interpolation: typing.Literal['nearest', 'bilinear'] | str = 'bilinear', edge_fill_value: float = 0.) -> torch.Tensor:
    # shape verification
    batch_size, n_channels, image_size, image_size = images.shape
    batch_size, _, _ = g_inv.shape

    # generate input coordinates
    in_coords_1d = torch.linspace(-1, 1, image_size, device=g_inv.device)                   # (image_size,)
    in_coords_i, in_coords_j = torch.meshgrid(in_coords_1d, in_coords_1d, indexing='ij')    # (image_size, image_size), (image_size, image_size)
    in_coords_3d = torch.stack([in_coords_i, in_coords_j,                                   # (image_size, image_size, 3)
        torch.ones_like(in_coords_i, device=g_inv.device)], dim=-1)

    out_coords_3d = (g_inv[..., None, None, :, :] @ in_coords_3d[..., None]).squeeze(-1)    # (batch_size, image_size, image_size, 3)
    out_coords_i, out_coords_j, _ = out_coords_3d.unbind(-1)                                # (batch_size, image_size, image_size), (batch_size, image_size, image_size)

    out_is = (out_coords_i[..., None, :, :] + 1) * ((image_size - 1) / 2)                   # (batch_size, 1, image_size, image_size)
    out_js = (out_coords_j[..., None, :, :] + 1) * ((image_size - 1) / 2)                   # (batch_size, 1, image_size, image_size)

    if interpolation == 'nearest':
        out_i0s = out_is.round().long()                                                         # (batch_size, 1, image_size, image_size)
        out_j0s = out_js.round().long()                                                         # (batch_size, 1, image_size, image_size)

        batch_indices = torch.arange(batch_size, device=images.device).reshape(-1, 1, 1, 1)     # (batch_size, 1, 1, 1)
        rgb_indices   = torch.arange(n_channels, device=images.device).reshape(1, -1, 1, 1)     # (1,          n_channels, 1, 1)
        out_i0s = out_i0s.clamp(0, image_size - 1)                                              # (batch_size, 1, image_size, image_size)
        out_j0s = out_j0s.clamp(0, image_size - 1)                                              # (batch_size, 1, image_size, image_size)

        out_images = images[batch_indices, rgb_indices, out_i0s, out_j0s]                       # (batch_size, n_channels, image_size, image_size)

    elif interpolation == 'bilinear':
        out_i0s = out_is.floor().long()                                                         # (batch_size, 1, image_size, image_size)
        out_j0s = out_js.floor().long()                                                         # (batch_size, 1, image_size, image_size)
        weight_i1 = out_is - out_i0s                                                            # (batch_size, 1, image_size, image_size)
        weight_j1 = out_js - out_j0s                                                            # (batch_size, 1, image_size, image_size)
        weight_i0 = 1 - weight_i1                                                               # (batch_size, 1, image_size, image_size)
        weight_j0 = 1 - weight_j1                                                               # (batch_size, 1, image_size, image_size)

        batch_indices = torch.arange(batch_size, device=images.device).reshape(-1, 1, 1, 1)     # (batch_size, 1, 1, 1)
        rgb_indices   = torch.arange(n_channels, device=images.device).reshape(1, -1, 1, 1)     # (1,          n_channels, 1, 1)
        out_i0s = out_i0s.clamp(0, image_size - 1)                                              # (batch_size, 1, image_size, image_size)
        out_j0s = out_j0s.clamp(0, image_size - 1)                                              # (batch_size, 1, image_size, image_size)
        out_i1s = (out_i0s + 1).clamp_max(image_size - 1)                                       # (batch_size, 1, image_size, image_size)
        out_j1s = (out_j0s + 1).clamp_max(image_size - 1)                                       # (batch_size, 1, image_size, image_size)

        out_images = (                                                                          # (batch_size, n_channels, image_size, image_size)
            weight_i0 * weight_j0 * images[batch_indices, rgb_indices, out_i0s, out_j0s] +
            weight_i1 * weight_j0 * images[batch_indices, rgb_indices, out_i1s, out_j0s] +
            weight_i0 * weight_j1 * images[batch_indices, rgb_indices, out_i0s, out_j1s] +
            weight_i1 * weight_j1 * images[batch_indices, rgb_indices, out_i1s, out_j1s])

    else:
        raise ValueError(f'Invalid interpolation mode: {interpolation}')

    out_images = out_images.where(
        (out_is >= 0) & (out_is <= image_size) &
        (out_js >= 0) & (out_js <= image_size),
        edge_fill_value)

    return out_images


