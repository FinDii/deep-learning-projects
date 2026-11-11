import numpy as np
import numpy.typing as npt
from skimage.exposure import rescale_intensity
from skimage.transform import resize
from scipy.ndimage import label
import torch

# 去除模型预测中的“小噪点”（假阳性小块）
def largest_connected_component(img: np.ndarray) -> npt.NDArray[np.bool_]:
    # from medpy.filter.binary.largest_connected_component
    # Returns the largest connected component of the input 2d-image as binary mask
    labeled_array, num_features = label(img, None)  # type: ignore
    component_sizes = np.bincount(labeled_array.ravel())[1:]
    return labeled_array == (np.argmax(component_sizes) + 1)

# 计算相似系数
def dsc(ys_pred, ys_true, lcc=True):
    # Computes dice similarity coefficient between imput images
    if lcc and np.any(ys_pred):
        ys_pred = np.round(ys_pred).astype(int)
        ys_true = np.round(ys_true).astype(int)
        ys_pred = largest_connected_component(ys_pred)
    return np.sum(ys_pred[ys_true == 1]) * 2.0 / (np.sum(ys_pred) + np.sum(ys_true))

# 将原始图像、真实掩码、预测掩码组合成可视化叠加图（红色轮廓=预测，绿色轮廓=真实）
def log_images(xs: torch.Tensor, y_true: torch.Tensor, y_pred: torch.Tensor, channel=1):
    images = []
    xs_np = xs[:, channel].cpu().numpy()
    ys_true_np = y_true[:, 0].cpu().numpy()
    ys_pred_np = y_pred[:, 0].cpu().numpy()
    for i in range(xs_np.shape[0]):
        image = gray2rgb(np.squeeze(xs_np[i]))
        image = outline(image, ys_pred_np[i], color=[255, 0, 0])
        image = outline(image, ys_true_np[i], color=[0, 255, 0])
        images.append(image)
    return images

# 归一化后，将单通道灰度图（0~1 浮点）转换为 3 通道 RGB 图像
def gray2rgb(image: npt.NDArray[np.floating]):
    w, h = image.shape
    image += np.abs(np.min(image))
    image_max = np.abs(np.max(image))
    if image_max > 0:
        image /= image_max

    ret = np.tile((image[..., None] * 255).astype(np.uint8), reps=3)
    return ret

# 在可视化中高亮预测或真实的边界，使轮廓清晰可见
def outline(image, mask, color):
    mask = np.round(mask)
    yy, xx = np.nonzero(mask)
    for y, x in zip(yy, xx):
        if 0.0 < np.mean(mask[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2]) < 1.0:
            image[max(0, y) : y + 1, max(0, x) : x + 1] = color
    return image

# 为 DataLoader 的每个子进程设置随机种子，确保数据加载的可重复性，避免多进程加载时产生重复序列
def dataloader_worker_init(worker_id):
    np.random.seed(42 + worker_id)


