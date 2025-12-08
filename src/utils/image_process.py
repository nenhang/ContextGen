import math
import random
from collections import defaultdict
from typing import List, Tuple, Optional, Union

import cv2
import PIL
import numpy as np
import supervision as sv
import torch
from PIL import Image
from heapq import heappush, heappop


def closest_upper_16x(n):
    return ((int(n) + 15) // 16) * 16


def closest_down_16x(n):
    return (int(n) // 16) * 16


def round_to_upper_16x(height, width):
    return closest_upper_16x(height), closest_upper_16x(width)


def normed_cxcywh_to_pixel_xyxy(bbox, width=1024, height=1024, return_dtype="tensor"):
    cx, cy, w, h = bbox
    cx_p, cy_p, w_p, h_p = (
        torch.round(cx * width),
        torch.round(cy * height),
        torch.round(w * width),
        torch.round(h * height),
    )
    cx_p = torch.clamp(cx_p, w_p / 2, width - w_p / 2)
    cy_p = torch.clamp(cy_p, h_p / 2, height - h_p / 2)
    x0 = torch.round(cx_p - w_p / 2).int()
    y0 = torch.round(cy_p - h_p / 2).int()
    x1 = x0 + w_p.int()
    y1 = y0 + h_p.int()
    x0 = torch.clamp(x0, 0, width - 1)
    y0 = torch.clamp(y0, 0, height - 1)
    x1 = torch.clamp(x1, 0, width - 1)
    y1 = torch.clamp(y1, 0, height - 1)
    if return_dtype == "tensor":
        bbox_pixel_xyxy = torch.stack([x0, y0, x1, y1], dim=0)
        bbox_normed_xyxy = torch.stack([x0 / width, y0 / height, x1 / width, y1 / height], dim=0)
    elif return_dtype == "list":
        bbox_pixel_xyxy = [x0.item(), y0.item(), x1.item(), y1.item()]
        bbox_normed_xyxy = [(x0 / width).item(), (y0 / height).item(), (x1 / width).item(), (y1 / height).item()]

    return bbox_pixel_xyxy, bbox_normed_xyxy


def get_default_height_width(
    image: Union[PIL.Image.Image, np.ndarray, torch.Tensor],
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> Tuple[int, int]:
    if height is None:
        if isinstance(image, PIL.Image.Image):
            height = image.height
        elif isinstance(image, torch.Tensor):
            height = image.shape[2]
        else:
            height = image.shape[1]

    if width is None:
        if isinstance(image, PIL.Image.Image):
            width = image.width
        elif isinstance(image, torch.Tensor):
            width = image.shape[3]
        else:
            width = image.shape[2]

    return height, width


def preprocess(
    self,
    image,
    height: Optional[int] = None,
    width: Optional[int] = None,
    resize_mode: str = "default",  # "default", "fill", "crop"
    crops_coords: Optional[Tuple[int, int, int, int]] = None,
) -> torch.Tensor:
    import warnings
    from diffusers.image_processor import is_valid_image_imagelist

    supported_formats = (PIL.Image.Image, np.ndarray, torch.Tensor)

    # Expand the missing dimension for 3-dimensional pytorch tensor or numpy array that represents grayscale image
    if self.config.do_convert_grayscale and isinstance(image, (torch.Tensor, np.ndarray)) and image.ndim == 3:
        if isinstance(image, torch.Tensor):
            # if image is a pytorch tensor could have 2 possible shapes:
            #    1. batch x height x width: we should insert the channel dimension at position 1
            #    2. channel x height x width: we should insert batch dimension at position 0,
            #       however, since both channel and batch dimension has same size 1, it is same to insert at position 1
            #    for simplicity, we insert a dimension of size 1 at position 1 for both cases
            image = image.unsqueeze(1)
        else:
            # if it is a numpy array, it could have 2 possible shapes:
            #   1. batch x height x width: insert channel dimension on last position
            #   2. height x width x channel: insert batch dimension on first position
            if image.shape[-1] == 1:
                image = np.expand_dims(image, axis=0)
            else:
                image = np.expand_dims(image, axis=-1)

    if isinstance(image, list) and isinstance(image[0], np.ndarray) and image[0].ndim == 4:
        warnings.warn(
            "Passing `image` as a list of 4d np.ndarray is deprecated."
            "Please concatenate the list along the batch dimension and pass it as a single 4d np.ndarray",
            FutureWarning,
        )
        image = np.concatenate(image, axis=0)
    if isinstance(image, list) and isinstance(image[0], torch.Tensor) and image[0].ndim == 4:
        warnings.warn(
            "Passing `image` as a list of 4d torch.Tensor is deprecated."
            "Please concatenate the list along the batch dimension and pass it as a single 4d torch.Tensor",
            FutureWarning,
        )
        image = torch.cat(image, axis=0)

    if not is_valid_image_imagelist(image):
        raise ValueError(
            f"Input is in incorrect format. Currently, we only support {', '.join(str(x) for x in supported_formats)}"
        )
    if not isinstance(image, list):
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        if crops_coords is not None:
            image = [i.crop(crops_coords) for i in image]
        if self.config.do_resize:  # don't resize here
            height, width = get_default_height_width(image[0], height, width)
            image = [self.resize(i, height, width, resize_mode=resize_mode) for i in image]
        if self.config.do_convert_rgb:
            image = [self.convert_to_rgb(i) for i in image]
        elif self.config.do_convert_grayscale:
            image = [self.convert_to_grayscale(i) for i in image]
        image = self.pil_to_numpy(image)  # to np
        image = self.numpy_to_pt(image)  # to pt

    elif isinstance(image[0], np.ndarray):
        image = np.concatenate(image, axis=0) if image[0].ndim == 4 else np.stack(image, axis=0)

        image = self.numpy_to_pt(image)

        height, width = get_default_height_width(image, height, width)
        if self.config.do_resize:
            image = self.resize(image, height, width)

    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, axis=0) if image[0].ndim == 4 else torch.stack(image, axis=0)

        if self.config.do_convert_grayscale and image.ndim == 3:
            image = image.unsqueeze(1)

        channel = image.shape[1]
        # don't need any preprocess if the image is latents
        if channel == self.config.vae_latent_channels:
            return image

        height, width = get_default_height_width(image, height, width)
        if self.config.do_resize:
            image = self.resize(image, height, width)

    # expected range [0,1], normalize to [-1,1]
    do_normalize = self.config.do_normalize
    if do_normalize and image.min() < 0:
        warnings.warn(
            "Passing `image` as torch tensor with value range in [-1,1] is deprecated. The expected value range for image tensor is [0,1] "
            f"when passing as pytorch tensor or numpy Array. You passed `image` with value range [{image.min()},{image.max()}]",
            FutureWarning,
        )
        do_normalize = False
    if do_normalize:
        image = self.normalize(image)

    if self.config.do_binarize:
        image = self.binarize(image)

    return image


def random_resize(input_size, max_long_edge=1280, min_short_edge=128):
    orig_w, orig_h = input_size
    stretch_factor = None if random.random() < 0.8 else random.uniform(1, 1.2)
    if stretch_factor is not None:  # 仅当需要拉伸时选择方向
        if random.random() < 0.5:
            stretched_w = orig_w * stretch_factor
            stretched_h = orig_h / stretch_factor
        else:
            stretched_h = orig_h * stretch_factor
            stretched_w = orig_w / stretch_factor
    else:
        stretched_w, stretched_h = orig_w, orig_h

    min_val = min(1, min_short_edge / min(stretched_w, stretched_h))
    max_val = max(1, max_long_edge / max(stretched_w, stretched_h))

    if min_val < max_val:
        ratio = (random.uniform(0, 1) ** 0.25) * (max_val - min_val) + min_val  # 强偏向高值
    else:
        ratio = 1.0

    stretched_w = stretched_w * ratio
    stretched_h = stretched_h * ratio

    return round_to_upper_16x(stretched_w, stretched_h)


def refine_bbox_16x(bbox, image_width, image_height) -> torch.Tensor:
    def refine_16x(x1, x2, image_width):
        if (x2 - x1) % 16 != 0:
            center_x = round((x1 + x2) / 2)
            width = closest_upper_16x(x2 - x1)
            assert width <= image_width, f"Width {width} exceeds image width {image_width}"
            x1 = max(0, center_x - width // 2)
            x1 = min(x1, image_width - width)
            x2 = x1 + width

        return x1, x2

    x1, y1, x2, y2 = bbox.tolist() if isinstance(bbox, torch.Tensor) else bbox
    x1, x2 = refine_16x(x1, x2, image_width)
    y1, y2 = refine_16x(y1, y2, image_height)
    return torch.tensor([x1, y1, x2, y2], dtype=torch.int32)


def uniform_resize_16x(input_size, base=512):
    w, h = input_size
    k = math.sqrt(base * base / (w * h))
    new_w = max(16, closest_upper_16x(w * k))
    new_h = max(16, closest_upper_16x(h * k))
    return new_w, new_h


def random_transform(image, mask=None):
    import albumentations as A

    if mask is not None:
        random_trans = A.Compose(
            [A.HorizontalFlip(p=0.2), A.Rotate(limit=10, p=0.2), A.ElasticTransform(p=0.2)],
            additional_targets={"image2": "image"},
        )
        transformed = random_trans(image=np.array(image), image2=np.array(mask))
        transformed_image = Image.fromarray(transformed["image"])
        transformed_mask = Image.fromarray(transformed["image2"])
        return transformed_image, transformed_mask
    else:
        random_trans = A.Compose([A.HorizontalFlip(p=0.2), A.Rotate(limit=10, p=0.2), A.ElasticTransform(p=0.2)])
        transformed = random_trans(image=np.array(image))
        transformed_image = Image.fromarray(transformed["image"])
        return transformed_image


def get_valid_mask_region(mask: torch.Tensor) -> tuple[int, int, int, int] | None:
    if mask.dtype != torch.bool:
        mask = mask.bool()
    if mask.dim() != 2:
        mask = mask.squeeze()
    assert mask.dim() == 2, "Mask must be 2D tensor (H,W)"

    # 检查全False的情况
    if not mask.any():
        return None

    # 计算非零区域边界
    rows = mask.any(dim=1)
    cols = mask.any(dim=0)

    y_indices = torch.where(rows)[0]
    x_indices = torch.where(cols)[0]

    y1, y2 = y_indices[[0, -1]] if len(y_indices) > 0 else (0, mask.shape[0])
    x1, x2 = x_indices[[0, -1]] if len(x_indices) > 0 else (0, mask.shape[1])

    return int(x1), int(y1), int(x2), int(y2)


def get_valid_mask_region_np(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)
    if mask.ndim != 2:
        mask = mask.squeeze()
    assert mask.ndim == 2, "Mask must be 2D array (H,W)"

    # 检查全False的情况
    if not np.any(mask):
        return None

    # 计算非零区域边界
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    y_indices = np.where(rows)[0]
    x_indices = np.where(cols)[0]

    y1, y2 = y_indices[[0, -1]] if len(y_indices) > 0 else (0, mask.shape[0])
    x1, x2 = x_indices[[0, -1]] if len(x_indices) > 0 else (0, mask.shape[1])

    return int(x1), int(y1), int(x2), int(y2)


def resize_to_fit_target_size(
    orig_size: tuple[int, int] | tuple[torch.Tensor, torch.Tensor],
    target_size: tuple[int, int] | tuple[torch.Tensor, torch.Tensor],
    max_resize_ratio: float | None = None,
    min_resize_ratio: float | None = None,
    fit_method: str = "min",
) -> tuple[int, int]:
    """
    Resize the image to fit the bounding box while maintaining aspect ratio.

    Args:
        target_size: Target size as a tuple (height, width) for the bounding box.
        orig_size: Original image size as a tuple (height, width).
        allow_distortion: If True, allows distortion to fit the target size exactly.
        fit_method: "min" to fit within the box, "max" to cover the box, "stretch" to stretch to exact size, "size" to resize to exact size without distortion.

    Returns:
        New image size as a tuple (new_height, new_width).
    """
    assert fit_method in ["min", "max", "stretch", "size"], "fit_method must be 'min', 'max', 'stretch' or 'size'"
    target_height, target_width = target_size
    original_height, original_width = orig_size

    if target_width <= 0 or target_height <= 0:
        raise ValueError("Invalid bounding box dimensions.")

    if fit_method == "stretch":
        new_height = target_height
        new_width = target_width
    else:
        if fit_method == "min":
            scale_factor = min(target_width / original_width, target_height / original_height)
        elif fit_method == "max":
            scale_factor = max(target_width / original_width, target_height / original_height)
        else:  # fit_method == "size"
            target_bbox_size = target_width * target_height
            original_size = original_width * original_height
            scale_factor = math.sqrt(target_bbox_size / original_size)

        if max_resize_ratio is not None:
            scale_factor = min(scale_factor, max_resize_ratio)
        if min_resize_ratio is not None:
            scale_factor = max(scale_factor, min_resize_ratio)

        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)

    return int(new_height), int(new_width)


def uniform_resize_advanced(
    input_size: tuple[int, int],  # should be in (height, width) format
    base_size: tuple[int, int],
    # 缩放返回限制
    max_resize_ratio: float | None = None,
    min_resize_ratio: float | None = None,
    restrict_16x: bool = False,
):
    h, w = input_size
    base_h, base_w = base_size
    # 计算缩放比例
    resize_ratio = math.sqrt(base_h * base_w / (h * w))
    # 限制缩放比例在指定范围内
    if max_resize_ratio is not None:
        resize_ratio = min(resize_ratio, max_resize_ratio)
    if min_resize_ratio is not None:
        resize_ratio = max(resize_ratio, min_resize_ratio)
    # 计算新的尺寸
    new_h = h * resize_ratio
    new_w = w * resize_ratio
    if restrict_16x:
        new_h = max(16, closest_down_16x(new_h))
        new_w = max(16, closest_down_16x(new_w))
    return int(new_h), int(new_w)


def uniform_resize_tensor_advanced_fit(
    input_size: tuple[int, int] | torch.Size,
    valid_part: tuple[int, int, int, int] | torch.Tensor,
    full_size: tuple[int, int] | tuple[torch.Tensor, torch.Tensor],
    bbox: tuple[int, int, int, int] | torch.Tensor,
) -> tuple[int, int, int, int]:
    # 转换为元组确保可索引
    if isinstance(input_size, torch.Size):
        input_size = tuple(input_size)
    if isinstance(full_size, torch.Tensor):
        full_size = tuple(full_size)

    # 解包尺寸
    input_h, input_w = input_size
    full_h, full_w = full_size

    # 计算输入中心点
    center_input_x = input_w // 2
    center_input_y = input_h // 2

    # 计算输入有效区域的偏移量
    vx1, vy1, vx2, vy2 = valid_part
    input_valid_part_offset = (vx1 - center_input_x, vy1 - center_input_y, vx2 - center_input_x, vy2 - center_input_y)

    # 计算目标框中心和尺寸
    bx1, by1, bx2, by2 = bbox
    bbox_center_x = (bx1 + bx2) // 2
    bbox_center_y = (by1 + by2) // 2
    bbox_w = bx2 - bx1
    bbox_h = by2 - by1

    # 计算调整偏移量
    resize_offset = (-bbox_center_x, -bbox_center_y, full_w - bbox_center_x, full_h - bbox_center_y)

    # 统一计算缩放比例 (使用生成器表达式提高效率)
    def calculate_positive_ratios(indices):
        ratios = []
        for i in indices:
            for j in indices:
                numerator = resize_offset[i]
                denominator = input_valid_part_offset[j]
                # 处理分母为零的情况
                if denominator == 0:
                    continue
                ratio = numerator / denominator
                if ratio > 0:
                    ratios.append(ratio)
        return ratios

    # 获取所有正数缩放比例
    x_ratios = calculate_positive_ratios([0, 2])  # 使用x方向索引
    y_ratios = calculate_positive_ratios([1, 3])  # 使用y方向索引
    all_positive_ratios = x_ratios + y_ratios

    # 处理没有有效缩放比例的情况
    if not all_positive_ratios:
        # 无法缩放时保持原尺寸
        scaling_ratio = 1.0
    else:
        # 使用最小正数比例确保所有约束
        scaling_ratio = min(all_positive_ratios)

    # 计算新尺寸
    new_h, new_w = resize_to_fit_target_size(
        orig_size=(input_h, input_w),
        target_size=(bbox_h, bbox_w),
        max_resize_ratio=scaling_ratio,
    )

    # 计算实际缩放比例
    actual_scale = new_h / input_h

    # 计算新的有效区域坐标
    def transform_coord(x, y):
        return (
            int((x - center_input_x) * actual_scale + bbox_center_x),
            int((y - center_input_y) * actual_scale + bbox_center_y),
        )

    new_v1 = transform_coord(vx1, vy1)
    new_v2 = transform_coord(vx2, vy2)

    return (*new_v1, *new_v2)


def resize_tensor_image(
    image: torch.Tensor,
    target_size: tuple[int, int],
):
    """
    Resize a tensor image to the target size using bilinear interpolation.

    Args:
        image: Input image tensor of shape (C, H, W) or (1, C, H, W).
        target_size: Target size as a tuple (height, width).

    Returns:
        Resized image tensor of shape (C, new_height, new_width) or (1, C, new_height, new_width).
    """
    assert image.dim() in [3, 4], "Image must be 3D (C, H, W) or 4D (1, C, H, W) tensor"
    if image.dim() == 4:
        assert image.size(0) == 1, "Batch size must be 1 for 4D tensor"
        input_dim = 4
    else:
        image = image.unsqueeze(0)  # Add batch dimension
        input_dim = 3

    target_size = (max(1, int(target_size[0])), max(1, int(target_size[1])))
    resized_image = torch.nn.functional.interpolate(image, size=target_size, mode="bilinear", align_corners=False)

    return resized_image if input_dim == 4 else resized_image.squeeze(0)


def stack_images_with_masks(
    images: list[torch.Tensor], masks: list[torch.Tensor], padding_value: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    叠加张量图像，利用bool类型mask控制可见区域（True为有效区域）
    下层图像会从上层无效区域中露出，无效部分最终置为灰色(输入中下层图像在前，上层图像在后)

    Args:
        images: 图像张量列表（相同尺寸），按从上到下顺序排列
        masks: 对应mask列表（bool类型），True表示有效区域

    Returns:
        stacked_img: 叠加后的RGB图像张量（无效区域为白色）
        final_mask: 叠加后的最终mask（有效区域为True）
    """
    input_image_dim = images[0].dim()
    input_mask_dim = masks[0].dim()

    images = [img.squeeze() for img in images]
    masks = [mask.squeeze() for mask in masks]
    assert all(img.dim() >= 2 for img in images), "Input must be tensors"
    assert all(mask.dtype == torch.bool for mask in masks), "Mask must be bool"

    device = images[0].device
    h, w = images[0].shape[-2:]

    stacked_img = torch.full((3, h, w), padding_value, dtype=images[0].dtype, device=device)
    final_mask = torch.zeros((h, w), dtype=torch.bool, device=device)

    for img, mask in zip(images, masks):
        stacked_img[:, mask] = img[:, mask]
        final_mask |= mask

    stacked_img = stacked_img.unsqueeze(0) if input_image_dim == 4 else stacked_img
    final_mask = (
        final_mask.unsqueeze(0).unsqueeze(0)
        if input_mask_dim == 4
        else (final_mask.unsqueeze(0) if input_mask_dim == 3 else final_mask)
    )

    return stacked_img, final_mask


def stack_images_with_alpha_blending(
    images: list[torch.Tensor],
    masks: list[torch.Tensor],
    alpha_min: float = 0.3,
    alpha_max: float = 0.8,
    padding_value: float = 0.0,  # 最终背景 (灰色, -1~1 归一化)
    blending_bg_value: float = 1.0,  # 混合时用来替代 padding_value 的值 (白色)
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    叠加张量图像，通过局部重叠数量和深度动态调整透明度。
    - Alpha 值完全基于每个像素的局部重叠组进行归一化。
    - 非重叠区域 (K=1) 的图像层将完全不透明 (alpha=1.0) 写入。

    Args:
        images: 图像张量列表（相同尺寸），按从下到上顺序排列。
        masks: 对应mask列表（bool类型），True表示有效区域。
        alpha_min: 重叠区域最下层图像的最小不透明度。
        alpha_max: 重叠区域最上层图像的最大不透明度。
        padding_value: 最终背景和初始混合背景值 (0.0 灰色)。
        blending_bg_value: 混合时用来替代 0.0 背景进行混合的值 (1.0 白色)。

    Returns:
        stacked_img: 叠加后的RGB图像张量（无效区域为 padding_value）
        final_mask: 叠加后的最终mask（有效区域为True）
    """
    if not images:
        return torch.empty(0), torch.empty(0)

    # --- 预处理和验证 ---
    input_image_dim = images[0].dim()
    input_mask_dim = masks[0].dim()

    images = [img.squeeze() for img in images]
    masks = [mask.squeeze() for mask in masks]

    def ensure_chw(img):
        img = img.squeeze()
        if img.dim() == 2:
            img = img.unsqueeze(0)
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        return img

    images = [ensure_chw(img) for img in images]

    device = images[0].device
    c, h, w = images[0].shape
    num_layers = len(images)

    # 将 masks 堆叠成 N x H x W 张量
    mask_tensor = torch.stack(masks, dim=0)

    # --- 阶段一：重叠和局部深度预计算 ---
    # 1. 计算每个像素的重叠图层数 (K)
    overlap_count = mask_tensor.long().sum(dim=0)  # H x W

    # 2. 计算每个有效像素的"局部深度" (j)
    # 局部深度 j: 对于一个有 K 个重叠的像素，j 从 0 (最底层) 变化到 K-1 (最上层)
    # 我们使用 N x H x W 的累积和来计算 j

    # cumsum(dim=0) 会在第一个维度上进行累加
    # 例：[[0, 1], [1, 1]] -> cumsum -> [[0, 1], [1, 2]]
    # j = cumsum - 1 (因为 j 从 0 开始)
    # (N x H x W)
    local_depth_j = mask_tensor.long().cumsum(dim=0) - 1

    # --- 阶段二：逐层动态混合 ---
    stacked_img = torch.full((c, h, w), padding_value, dtype=images[0].dtype, device=device)
    pixels_covered = torch.zeros((h, w), dtype=torch.bool, device=device)
    final_mask = overlap_count > 0  # 最终有效区域

    alpha_delta = alpha_max - alpha_min

    for i in range(num_layers):
        img = images[i]
        mask = masks[i]

        # 1. 识别三种区域的局部掩码
        # 区域 A: 非重叠区域 (K=1)
        is_single_layer = (overlap_count == 1) & mask

        # 区域 B: 重叠区域 (K>1)
        is_overlap_layer = (overlap_count > 1) & mask

        # 2. 处理 非重叠区域 (区域 A): 完全不透明写入 (alpha=1.0)
        if is_single_layer.any():
            stacked_img[:, is_single_layer] = img[:, is_single_layer]

        # 3. 处理 重叠区域 (区域 B): 动态 Alpha 混合
        if is_overlap_layer.any():
            # 获取局部重叠计数 K 和局部深度 j，仅在 is_overlap_layer 区域
            # K_local 和 j_local 仍然是 H x W 张量，但我们只关心 is_overlap_layer 区域的值
            K_local = overlap_count[is_overlap_layer].float()
            j_local = local_depth_j[i][is_overlap_layer].float()

            # --- 关键修正：计算局部归一化因子和 Alpha ---
            # K-1 作为归一化因子，避免 K=1 时除以 0 (K>1 区域保证 K-1 >= 1)
            norm_factor_local = torch.clamp(K_local - 1.0, min=1.0)

            # 局部归一化深度： j / (K - 1)
            normalized_depth = j_local / norm_factor_local

            # 局部动态 Alpha： alpha_min + Delta * (j / (K - 1))
            # current_alpha 是一个 (N_overlap) 长度的向量，每个像素有自己的 alpha 值
            current_alpha_vec = alpha_min + alpha_delta * normalized_depth

            # 将 alpha 扩展到 C 维度，用于图像乘法
            current_alpha_vec_expanded = current_alpha_vec.unsqueeze(0).repeat(c, 1)

            # --- 准备混合背景 C_bg ---
            C_bg_temp = stacked_img.clone()

            # 识别需要替换背景的区域：当前重叠 mask=True, 且从未被任何层覆盖 (pixels_covered=False)
            uncovered_in_overlap_mask = is_overlap_layer & (~pixels_covered)

            # 在这些首次被覆盖的区域，将 C_bg_temp 中的 0.0 临时替换为 1.0
            C_bg_temp[:, uncovered_in_overlap_mask] = blending_bg_value

            # --- 执行 Alpha 混合 ---
            current_img_area = img[:, is_overlap_layer]
            C_bg_area = C_bg_temp[:, is_overlap_layer]

            # Alpha 混合公式：C_out = alpha(p) * C_fg(p) + (1 - alpha(p)) * C_bg(p)
            blended_area = (
                current_alpha_vec_expanded * current_img_area + (1.0 - current_alpha_vec_expanded) * C_bg_area
            )

            # 将混合结果写回 stacked_img
            stacked_img[:, is_overlap_layer] = blended_area

        # 4. 更新覆盖标记
        pixels_covered |= mask

    # --- 阶段三：最终背景处理 ---
    # 将 ~final_mask 区域设回 padding_value (0.0 灰色)
    # final_mask 是 (H x W)，需要扩展到 C x H x W
    final_mask_expanded = final_mask.unsqueeze(0).repeat(c, 1, 1)
    stacked_img[~final_mask_expanded] = padding_value

    # --- 恢复原始维度结构 ---
    stacked_img = stacked_img.unsqueeze(0) if input_image_dim == 4 else stacked_img

    if input_mask_dim == 4:
        final_mask = final_mask.unsqueeze(0).unsqueeze(0)
    elif input_mask_dim == 3:
        final_mask = final_mask.unsqueeze(0)

    return stacked_img, final_mask


def expand_to_full_image_size(
    image: torch.Tensor,
    full_size: tuple[int, int],
    center: tuple[int, int],
    padding_value: int | float | bool,
    loc_point: tuple[tuple[float, float], tuple[float, float]] | None = None,
):
    """
    将图像放置在指定中心位置，并把周围置为 0

    Args:
        image: 输入图像张量 (C, H, W) 或 (1, C, H, W)
        full_size: 目标全图像大小 (height, width)
        center: 图像中心位置 (y, x)
        padding_value: 填充颜色，默认为白色 (255, 255, 255),
        loc_point: 可选的定位点，格式为归一化后的 ((source_y, source_x), (target_y, target_x))

    Returns:
        expanded_image: 扩展后的图像张量，填充为白色背景
    """
    assert image.dim() in [3, 4], "Image must be 3D (C, H, W) or 4D (1, C, H, W) tensor"
    if image.dim() == 4:
        assert image.size(0) == 1, "Batch size must be 1 for 4D tensor"
        input_dim = 4
        image = image.squeeze(0)
    else:
        input_dim = 3

    full_height, full_width = full_size
    img_height, img_width = image.shape[1:3]

    # 创建全白背景
    expanded_image = torch.full(
        (image.size(0), full_height, full_width), padding_value, dtype=image.dtype, device=image.device
    )

    # 计算放置位置
    if loc_point is None:
        src_y, src_x = img_height // 2, img_width // 2
        tgt_y, tgt_x = center
    else:
        (src_y, src_x), (tgt_y, tgt_x) = loc_point
        if isinstance(src_y, float):
            src_y = int(round(src_y * img_height))
            src_x = int(round(src_x * img_width))
            tgt_y = int(round(tgt_y * full_height))
            tgt_x = int(round(tgt_x * full_width))
        elif isinstance(src_y, torch.Tensor):
            src_y = torch.round(src_y * img_height).int().item()
            src_x = torch.round(src_x * img_width).int().item()
            tgt_y = torch.round(tgt_y * full_height).int().item()
            tgt_x = torch.round(tgt_x * full_width).int().item()

    # 计算图像在全尺寸图像中的起始位置
    start_y = tgt_y - src_y
    start_x = tgt_x - src_x

    # 确定有效的放置区域和对应的图像切片
    # 目标区域 (在 expanded_image 中)
    paste_y_start = max(0, start_y)
    paste_x_start = max(0, start_x)
    paste_y_end = min(full_height, start_y + img_height)
    paste_x_end = min(full_width, start_x + img_width)

    # 对应源图像切片 (在 image 中)
    image_y_start = max(0, -start_y)
    image_x_start = max(0, -start_x)
    image_y_end = image_y_start + (paste_y_end - paste_y_start)
    image_x_end = image_x_start + (paste_x_end - paste_x_start)

    # 将图像放置在指定位置
    expanded_image[:, paste_y_start:paste_y_end, paste_x_start:paste_x_end] = image[
        :, image_y_start:image_y_end, image_x_start:image_x_end
    ]

    return expanded_image if input_dim == 3 else expanded_image.unsqueeze(0)


def determine_overlap(
    mask1: torch.Tensor,
    mask2: torch.Tensor,
) -> dict:
    """
    计算两个 mask 之间的详细重叠信息。
    :param mask1: 第一个 mask (H,W) 或 (C,H,W) 或 (1,C,H,W)，bool类型
    :param mask2: 第二个 mask (H,W) 或 (C,H,W) 或 (1,C,H,W)，bool类型
    :return: 包含重叠信息的字典:
             - 'relation': 1 (mask1包含mask2), -1 (mask2包含mask1), 0 (无完全包含)
             - 'iou': 两个mask的Jaccard相似系数 (IoU)
             - 'area1': mask1的面积 (True像素数)
             - 'area2': mask2的面积 (True像素数)
    """
    if mask1.dim() > 2:
        mask1 = mask1.squeeze()
    if mask2.dim() > 2:
        mask2 = mask2.squeeze()
    assert mask1.dim() == mask2.dim() == 2, "Masks must be 2D tensors"
    assert mask1.dtype == mask2.dtype == torch.bool, "Masks must be bool type"

    if mask1.shape != mask2.shape:
        raise ValueError("Masks must have the same shape")

    # 计算交集和并集
    intersection = (mask1 & mask2).sum().item()
    union = (mask1 | mask2).sum().item()

    area1 = mask1.sum().item()
    area2 = mask2.sum().item()

    iou = intersection / union if union > 0 else 0.0

    relation = 0
    if area1 > 0 and area2 > 0:  # 避免空mask导致的包含判断错误
        # 检查 mask1 是否完全包含 mask2 (即 mask2 被 mask1 包裹)
        is_mask2_contained_in_mask1 = torch.all(mask1[mask2])
        # 检查 mask2 是否完全包含 mask1 (即 mask1 被 mask2 包裹)
        is_mask1_contained_in_mask2 = torch.all(mask2[mask1])

        if is_mask1_contained_in_mask2:  # mask1 完全包含在 mask2 中
            relation = 1
        elif is_mask2_contained_in_mask1:  # mask2 完全包含在 mask1 中
            relation = -1

    return {"relation": relation, "iou": iou, "area1": area1, "area2": area2}


def determine_mask_sequence(masks: list[torch.Tensor]) -> list[int]:
    """
    基于包含关系、面积比和重叠情况选择性随机的动态排序算法。
    1. 构建完全包含关系有向图 (A→B表示A包含B)。
    2. 收集所有部分重叠的mask对信息。
    3. 使用拓扑排序，并在优先级队列中根据重叠情况选择不同的优先级生成策略。
    """
    n = len(masks)
    if n == 0:
        return []
    if n == 1:
        return [0]

    graph = defaultdict(set)
    in_degree = [0] * n
    # 存储部分重叠的对，键是节点索引，值是与其重叠的 (other_node_idx, iou, other_area) 列表
    overlap_relationships = defaultdict(list)

    all_mask_areas = [mask.sum().item() for mask in masks]

    for i in range(n):
        for j in range(i + 1, n):
            area_i = all_mask_areas[i]
            area_j = all_mask_areas[j]

            # 过滤掉两个都是空mask的情况，或一个空mask与非空mask的情况
            if area_i == 0 and area_j == 0:
                continue
            if area_i == 0 or area_j == 0:
                continue  # 空mask不参与包含/重叠判断，留给最后随机排序

            rel_data = determine_overlap(masks[i], masks[j])
            relation = rel_data["relation"]
            iou = rel_data["iou"]

            # a -> b 表示 a 包含 b
            if relation == 1:  # mask_i 被 mask_j 完全包含，则到时候要先画 j 再画 i
                graph[j].add(i)
                in_degree[i] += 1
            elif relation == -1:
                graph[i].add(j)
                in_degree[j] += 1
            elif iou > 0:  # 部分重叠 (IoU > 0 且无完全包含关系)
                overlap_relationships[i].append((j, iou, area_j))
                overlap_relationships[j].append((i, iou, area_i))

    # 辅助函数：根据重叠情况计算优先级
    def calculate_priority(node_idx):
        node_area = all_mask_areas[node_idx]

        # 检查当前节点是否有部分重叠关系
        has_partial_overlap = len(overlap_relationships[node_idx]) > 0

        if not has_partial_overlap:
            return random.random()
        else:
            total_probability_to_be_lower = 0
            total_obscured_iou = 0

            for other_node_idx, iou, other_area in overlap_relationships[node_idx]:
                if node_area == 0 or other_area == 0:
                    continue
                # 计算与这个特定重叠 Mask 的倾向性概率
                prob_lower_for_pair = node_area / (node_area + other_area)

                # 权重：IoU 越高，该对重叠的影响越大
                total_probability_to_be_lower += prob_lower_for_pair * iou
                total_obscured_iou += iou  # 使用IoU作为权重来计数，或简单 count_overlaps += 1

            if total_obscured_iou > 0:
                avg_probability_to_be_lower = total_probability_to_be_lower / total_obscured_iou
            else:  # 只有 IoU=0 的重叠关系，或者都是空 Mask 导致没计算出有效概率
                avg_probability_to_be_lower = 0.5  # 默认中性

            scaled_total_obscured_iou = total_obscured_iou * 0.1  # 0.1 为经验值，调节 IoU 对优先级的影响
            # avg_probability_to_be_lower 越大，scaled_total_obscured_iou 越大，表示该实例越有可能被遮挡，应该降低优先级，推迟出队
            combined_probability_to_be_lower = avg_probability_to_be_lower + scaled_total_obscured_iou

            average_iou = total_probability_to_be_lower / total_obscured_iou if total_obscured_iou > 0 else 0.0
            range_width_factor = max(0.0, 1.0 - average_iou * 0.8)  # 0.8 是强度调节因子

            # 越有可能被遮挡的越迟出队
            center_priority = 1.0 - combined_probability_to_be_lower

            random_offset_max = 0.2  # 最大随机偏移量，可调
            random_offset = random.uniform(-random_offset_max, random_offset_max) * range_width_factor

            final_priority = center_priority + random_offset
            final_priority = max(0.0, min(1.0, final_priority))

            return final_priority

    # 3. 拓扑排序
    heap = []

    # 初始入队：所有入度为0且非空的节点
    for node in range(n):
        if in_degree[node] == 0 and all_mask_areas[node] > 0:
            heappush(heap, (calculate_priority(node), node))
        # 空 Mask 节点不会进入这个堆，它们会在最后的 remaining_nodes 中被处理

    topo_order = []
    visited = set()

    while heap:
        current_priority, node = heappop(heap)
        if node in visited:
            continue
        visited.add(node)

        topo_order.append(node)

        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0 and neighbor not in visited:
                if all_mask_areas[neighbor] > 0:  # 确保非空 Mask 才能被添加到堆中
                    heappush(heap, (calculate_priority(neighbor), neighbor))
                else:
                    visited.add(neighbor)

    # 处理剩余的无影响节点 (包括所有空 Mask，以及没有包含关系且没有有效重叠影响的 Mask)
    remaining_nodes = [node for node in range(n) if node not in visited]
    random.shuffle(remaining_nodes)
    topo_order.extend(remaining_nodes)

    return topo_order


def place_instance_image_and_mask(
    instance_image: torch.Tensor,
    instance_mask: torch.Tensor,
    bbox: torch.Tensor,
    full_size: tuple[int, int],
    resize_pattern: str = "instance-fit",
    valid_part_bbox: torch.Tensor | list | tuple | None = None,
    loc_point: tuple[tuple[int, int], tuple[int, int]] | None = None,
    using_dino: bool = False,
    dino_info: dict | None = None,
    device: str | torch.device = "cuda",
):
    assert resize_pattern in ["instance-fit", "instance-fit-distort", "image-fit"], (
        "resize_pattern must be 'instance_fit' or 'image_fit'."
    )
    input_size = instance_image.shape[-2:]

    instance_mask_bool = instance_mask > 0
    valid_instance_bounding_box = get_valid_mask_region(instance_mask_bool)

    if valid_instance_bounding_box is None:
        return torch.zeros(
            (instance_image.shape[0], instance_image.shape[1], full_size[0], full_size[1]),
            dtype=instance_image.dtype,
            device=instance_image.device,
        ), torch.zeros(
            (instance_mask.shape[0], instance_mask.shape[1], full_size[0], full_size[1]),
            dtype=torch.bool,
            device=instance_mask.device,
        )

    if valid_part_bbox is not None:
        if isinstance(valid_part_bbox, (list, tuple)):
            valid_part_bbox = torch.tensor(valid_part_bbox, dtype=torch.float32, device=device)
        assert isinstance(valid_part_bbox, torch.Tensor)
        valid_part_bbox = valid_part_bbox * torch.tensor(
            [input_size[1], input_size[0], input_size[1], input_size[0]], dtype=torch.float32, device=device
        )
        valid_part_bbox = valid_part_bbox.int()
        x1 = min(valid_part_bbox[0], valid_instance_bounding_box[0])
        y1 = min(valid_part_bbox[1], valid_instance_bounding_box[1])
        x2 = max(valid_part_bbox[2], valid_instance_bounding_box[2])
        y2 = max(valid_part_bbox[3], valid_instance_bounding_box[3])
    elif not using_dino:
        x1, y1, x2, y2 = valid_instance_bounding_box
    else:
        from groundingdino.util.inference import load_image as dino_load_image, predict

        assert "model" in dino_info and "image" in dino_info and "prompt" in dino_info, (
            "make sure dino_info contains 'dino_model', 'image' and 'prompt' keys."
        )
        dino_model = dino_info["model"]
        dino_image = dino_info["image"]
        dino_prompt = dino_info["prompt"]
        dino_device = dino_info.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        dino_image_source, dino_image = dino_load_image(dino_image)
        if not dino_prompt.endswith("."):
            dino_prompt += " ."
        with torch.inference_mode():
            dino_bboxes = predict(
                model=dino_model,
                image=dino_image,
                caption=dino_prompt,
                box_threshold=0.35,
                text_threshold=0.25,
                device=dino_device,
            )[0]
        if dino_bboxes is None or len(dino_bboxes) == 0:
            x1, y1, x2, y2 = valid_instance_bounding_box
        else:
            x1, y1, x2, y2 = normed_cxcywh_to_pixel_xyxy(
                dino_bboxes[0], width=dino_image_source.shape[1], height=dino_image_source.shape[0]
            )[0]

    instance_image = instance_image[:, :, y1:y2, x1:x2]
    instance_mask = instance_mask[:, :, y1:y2, x1:x2]
    if resize_pattern == "instance-fit":
        new_size = resize_to_fit_target_size(
            target_size=(bbox[3] - bbox[1], bbox[2] - bbox[0]), orig_size=(y2 - y1, x2 - x1), fit_method="size"
        )
        bbox_center = int((bbox[1] + bbox[3]) // 2), int((bbox[0] + bbox[2]) // 2)
    elif resize_pattern == "instance-fit-distort":
        new_size = resize_to_fit_target_size(
            target_size=(bbox[3] - bbox[1], bbox[2] - bbox[0]), orig_size=(y2 - y1, x2 - x1), fit_method="stretch"
        )
        bbox_center = int((bbox[1] + bbox[3]) // 2), int((bbox[0] + bbox[2]) // 2)
    elif resize_pattern == "image-fit":
        new_valid_bbox = uniform_resize_tensor_advanced_fit(
            input_size=input_size, valid_part=[x1, y1, x2, y2], full_size=full_size, bbox=bbox
        )
        new_size = int(new_valid_bbox[3] - new_valid_bbox[1]), int(new_valid_bbox[2] - new_valid_bbox[0])
        bbox_center = (
            int((new_valid_bbox[1] + new_valid_bbox[3]) // 2),
            int((new_valid_bbox[0] + new_valid_bbox[2]) // 2),
        )

    instance_image = resize_tensor_image(instance_image, new_size)
    instance_mask = resize_tensor_image(instance_mask, new_size)
    instance_mask = instance_mask > 0
    instance_mask = expand_to_full_image_size(
        instance_mask, full_size, bbox_center, padding_value=False, loc_point=loc_point
    )
    instance_image = expand_to_full_image_size(
        instance_image, full_size, bbox_center, padding_value=0.0, loc_point=loc_point
    )
    return instance_image, instance_mask


def annotate(image_source: Image.Image, boxes: List[torch.Tensor], phrases: List[str]) -> Image.Image:
    xyxy = torch.stack(boxes, dim=0).cpu().numpy()
    detections = sv.Detections(xyxy=xyxy)

    labels = [f"{phrase}" for phrase in phrases]

    bbox_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
    label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
    annotated_frame = image_source.copy()
    annotated_frame = bbox_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    return annotated_frame


def annotate_cv2(
    image: Image.Image,
    boxes: List[List[int]],  # [x1, y1, x2, y2]
    phrases: List[str],
    box_colors: List[Tuple[int, int, int]],  # RGB格式，如 (255,0,0)
    text_colors: List[Tuple[int, int, int]] = [(255, 255, 255)],  # 默认白色
    font_scale: float = 0.8,
    thickness: int = 2,
    margin: int = 5,  # 标签与框的最小外边距
) -> Image.Image:
    # PIL转OpenCV格式（RGB->BGR）
    img_np = np.array(image)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_h = img_np.shape[0]

    for i, (box, phrase) in enumerate(zip(boxes, phrases)):
        x1, y1, x2, y2 = map(int, box)
        color = tuple(box_colors[i % len(box_colors)][::-1])  # RGB->BGR
        text_color = tuple(text_colors[i % len(text_colors)][::-1])  # RGB->BGR

        # 画检测框
        cv2.rectangle(img_np, (x1, y1), (x2, y2), color, thickness)

        # 计算文本所需空间
        (text_w, text_h), _ = cv2.getTextSize(phrase, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        # 判断标签位置
        if y1 >= text_h + margin:  # 上方有足够空间
            # 放在框外顶部
            label_y1 = y1 - text_h - margin
            label_y2 = y1
            text_y = y1 - margin // 2
        else:
            # 放在框内左上角
            label_y1 = y1 + margin
            label_y2 = y1 + text_h + margin
            text_y = y1 + text_h + margin // 2

        # 画标签背景（颜色与框相同）
        cv2.rectangle(
            img_np,
            (x1, label_y1),
            (x1 + text_w, label_y2),
            color,
            -1,  # -1表示填充
        )

        # 写文字
        cv2.putText(img_np, phrase, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

    # 转回PIL格式（BGR->RGB）
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_np)


def annotate_two_halves(
    image_source: Image.Image, boxes: List[torch.Tensor], phrases: List[str], pattern: str = "left-right"
) -> Image.Image:
    assert pattern in ["up-down", "left-right"], "Pattern must be either 'up-down' or 'left-right'."
    # 如果是 up-down，则将图片等分为上下两部分，分别标注完之后拼合起来
    if pattern == "up-down":
        half_height = image_source.height // 2
        top_half = image_source.crop((0, 0, image_source.width, half_height))
        bottom_half = image_source.crop((0, half_height, image_source.width, image_source.height))
        top_annotated = annotate(top_half, boxes, phrases)
        bottom_annotated = annotate(bottom_half, boxes, phrases)
        annotated_image = Image.new("RGB", (image_source.width, image_source.height))
        annotated_image.paste(top_annotated, (0, 0))
        annotated_image.paste(bottom_annotated, (0, half_height))
    else:  # left-right
        half_width = image_source.width // 2
        left_half = image_source.crop((0, 0, half_width, image_source.height))
        right_half = image_source.crop((half_width, 0, image_source.width, image_source.height))
        left_annotated = annotate(left_half, boxes, phrases)
        right_annotated = annotate(right_half, boxes, phrases)
        annotated_image = Image.new("RGB", (image_source.width, image_source.height))
        annotated_image.paste(left_annotated, (0, 0))
        annotated_image.paste(right_annotated, (half_width, 0))
    return annotated_image


def get_union_of_masks(masks: List[torch.Tensor] | torch.Tensor) -> torch.Tensor:
    if isinstance(masks, torch.Tensor):
        masks = [masks]
    assert all(mask.dim() in [2, 3, 4] for mask in masks), "Masks must be 2D, 3D or 4D tensors"

    union_mask = torch.zeros_like(masks[0], dtype=torch.bool)
    for mask in masks:
        union_mask |= mask.squeeze() if mask.dim() > 2 else mask

    return union_mask.unsqueeze(0) if union_mask.dim() == 2 else union_mask
