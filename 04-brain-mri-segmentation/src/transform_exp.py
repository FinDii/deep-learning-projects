
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter


def elastic_transform(image, mask, alpha, sigma, random_state=None):
    """
    对图像和掩码同时应用弹性形变，保证一致性
    image: (H, W, C) 或 (H, W)
    mask:  (H, W) 或 (H, W, 1)
    """
    if random_state is None:
        random_state = np.random.RandomState(None)
    shape = image.shape[:2]  # (H, W)
    dx = gaussian_filter(random_state.randn(*shape), sigma, mode='constant', cval=0) * alpha
    dy = gaussian_filter(random_state.randn(*shape), sigma, mode='constant', cval=0) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), indexing='xy')
    indices = (y + dy, x + dx)

    # 处理图像
    if image.ndim == 3:
        warped_image = np.zeros_like(image)
        for c in range(image.shape[2]):
            warped_image[..., c] = map_coordinates(image[..., c], indices, order=1, mode='reflect')
    else:
        warped_image = map_coordinates(image, indices, order=1, mode='reflect')
    
    # 处理掩码 (最近邻插值保持二值性)
    if mask.ndim == 3:
        warped_mask = np.zeros_like(mask)
        for c in range(mask.shape[2]):
            warped_mask[..., c] = map_coordinates(mask[..., c], indices, order=0, mode='reflect')
    else:
        warped_mask = map_coordinates(mask, indices, order=0, mode='reflect')
    
    return warped_image, warped_mask


class ElasticTransform:
    def __init__(self, alpha=34, sigma=4, p=0.5):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def __call__(self, image, mask):
        if np.random.random() < self.p:
            return elastic_transform(image, mask, self.alpha, self.sigma)
        return image, mask
