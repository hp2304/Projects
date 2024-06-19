from copy import deepcopy
from typing import List
from configs import transform_mat, ref_vec
import numpy as np


def func(u: np.ndarray) -> np.ndarray:
    a = u ** (1 / 3)
    b = (16 / 116) + (7.787 * u)
    mask = (u > 0.008856).astype(np.float32)
    return (mask * a) + ((1 - mask) * b)


def compute_l_star(pixels: np.ndarray) -> np.ndarray:
    common_y = (pixels[:, 1] / ref_vec[0, 1])
    common_y_powered = common_y ** (1 / 3)
    a = (116 * common_y_powered) - 16
    b = 903.3 * common_y

    mask = (common_y > 0.008856).astype(np.float32)
    return (mask * a) + ((1 - mask) * b)


def compute_a_star(pixels: np.ndarray) -> np.ndarray:
    common_x = (pixels[:, 0] / ref_vec[0, 0])
    common_y = (pixels[:, 1] / ref_vec[0, 1])
    return 500 * (func(common_x) - func(common_y))


def clamp_signed_8bit(data: np.ndarray) -> np.ndarray:
    data[data < -128] = -128
    data[data > 127] = 127
    return data + 128


def compute_b_star(pixels: np.ndarray) -> np.ndarray:
    common_x = (pixels[:, 0] / ref_vec[0, 0])
    common_z = (pixels[:, 2] / ref_vec[0, 2])
    return 200 * (func(common_x) - func(common_z))


def quantize_lab(img_lab: np.ndarray,
                 lab_bins: List[int] = None) -> np.ndarray:
    img_lab_q = deepcopy(img_lab)
    if lab_bins is None or len(lab_bins) != 3:
        lab_bins = [10, 3, 3]

    img_lab_q[:, :, 1] = clamp_signed_8bit(img_lab_q[:, :, 1])
    img_lab_q[:, :, 2] = clamp_signed_8bit(img_lab_q[:, :, 2])

    q_l = lambda v, bins: v // (101 / bins)
    q_ab = lambda v, bins: v // (256 / bins)
    for i, bins in enumerate(lab_bins):
        if i == 0:
            img_lab_q[:, :, i] = q_l(img_lab_q[:, :, i], bins)
        else:
            img_lab_q[:, :, i] = q_ab(img_lab_q[:, :, i], bins)

    # this corresponds to binning [10, 3, 3]
    lab_pixels_encoded = (9 * img_lab_q[:, :, 0]) + (3 * img_lab_q[:, :, 1]) + img_lab_q[:, :, 2]
    return lab_pixels_encoded.astype(np.int32)


def rgb_to_lab(rgb_pixels: np.ndarray) -> np.ndarray:
    rgb_pixels = np.matmul(transform_mat, rgb_pixels).T

    l_star = compute_l_star(rgb_pixels)
    a_star = compute_a_star(rgb_pixels)
    b_star = compute_b_star(rgb_pixels)

    return np.array([l_star, a_star, b_star]).T
