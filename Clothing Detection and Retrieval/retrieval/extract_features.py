import time

import cv2
import numpy as np
from configs import LAB_NB_BINS, NB_ANGLE_BINS
from color_conversion import rgb_to_lab, quantize_lab
from grad_calculation import compute_orientation, quantize_angle


def extract_neighbors(data: np.ndarray, dist: int = 1) -> np.ndarray:
    # data has shape (m, n, f). m and n refers to spatial dimensions and f refers to feature dimension
    data_copy = np.pad(data, ((dist, dist), (dist, dist), (0, 0)), constant_values=-1).copy()
    h, w, f = data_copy.shape
    nb_elements_in_window = (2 * dist + 1) ** 2
    mid_element_idx = nb_elements_in_window // 2
    neighbors = []
    for i in range(dist, h - dist):
        for j in range(dist, w - dist):
            window = data_copy[i - dist: i + dist + 1, j - dist: j + dist + 1].copy().reshape(nb_elements_in_window, f)
            window = np.delete(window, mid_element_idx, axis=0)
            neighbors.append(window)

    return np.array(neighbors)


def extract_features_vectorized(img_lab: np.ndarray,
                                color_enc: np.ndarray,
                                angle_enc: np.ndarray,
                                color_f_len: int,
                                angle_f_len: int) -> np.ndarray:
    h, w = img_lab.shape[:2]
    img_lab_flattened = img_lab.reshape(h * w, 1, 3)
    neighbor_lab_flattened = extract_neighbors(img_lab)

    lab_diff_sum_sqrt = np.sqrt(((img_lab_flattened - neighbor_lab_flattened) ** 2).sum(axis=2))

    color_enc_flattened = color_enc.reshape(h * w, 1, 1)
    neighbor_color_enc_flattened = extract_neighbors(color_enc[..., np.newaxis])

    angle_enc_flattened = angle_enc.reshape(h * w, 1, 1)
    neighbor_angle_enc_flattened = extract_neighbors(angle_enc[..., np.newaxis])

    mask = (color_enc_flattened == neighbor_color_enc_flattened) & (angle_enc_flattened == neighbor_angle_enc_flattened)
    mask = mask.astype(np.float32).squeeze()
    lab_diff_collected = (lab_diff_sum_sqrt * mask).sum(axis=1)

    angle_features = np.zeros(angle_f_len, dtype=np.float32)
    color_features = np.zeros(color_f_len, dtype=np.float32)
    for i in range(h * w):
        color_id = color_enc_flattened[i, 0, 0]
        angle_id = angle_enc_flattened[i, 0, 0]
        angle_features[angle_id] += lab_diff_collected[i]
        color_features[color_id] += lab_diff_collected[i]
    features = np.concatenate([color_features, angle_features])
    return features / 2


def extract_features_naive(img: np.ndarray,
                           color_enc: np.ndarray,
                           angle_enc: np.ndarray,
                           color_f_len: int,
                           angle_f_len: int,
                           dist_th: int = 1) -> np.ndarray:
    h, w = color_enc.shape

    # assuming elements of data1 are less than features_len
    angle_features = np.zeros(angle_f_len, dtype=np.float32)
    color_features = np.zeros(color_f_len, dtype=np.float32)
    for i in range(h):
        for j in range(w):
            center_color_enc = color_enc[i, j]
            center_angle_enc = angle_enc[i, j]
            for neighbor_i in range(i - dist_th, i + dist_th + 1):
                for neighbor_j in range(j - dist_th, j + dist_th + 1):
                    # ignore out of bounds neighbors and center pixel
                    if (neighbor_i == i and neighbor_j == j) or (neighbor_i < 0 or neighbor_i >= h) or (
                            neighbor_j < 0 or neighbor_j >= w):
                        continue
                    neighbor_color_enc = color_enc[neighbor_i, neighbor_j]
                    neighbor_angle_enc = angle_enc[neighbor_i, neighbor_j]

                    if center_color_enc == neighbor_color_enc and center_angle_enc == neighbor_angle_enc:
                        color_diff = 0

                        # NOTE: assuming number of channels are 3
                        for channel in range(3):
                            color_diff += (img[i, j, channel] - img[neighbor_i, neighbor_j, channel]) ** 2
                        color_diff = np.sqrt(color_diff)
                        color_features[center_color_enc] += color_diff
                        angle_features[center_angle_enc] += color_diff

    features = np.concatenate([color_features, angle_features])
    return features / 2


def get_bottom_and_right_neighbor_directions(dist: int):
    directions = []

    # right neighbors
    for j in range(1, dist + 1):
        directions.append([0, j])

    # bottom neighbors
    for i in range(1, dist + 1):
        for j in range(-dist, dist + 1):
            directions.append([i, j])

    return directions


def extract_features_optimized(img: np.ndarray,
                               color_enc: np.ndarray,
                               angle_enc: np.ndarray,
                               color_f_len: int,
                               angle_f_len: int,
                               dist_th: int = 1) -> np.ndarray:
    h, w = color_enc.shape
    neighbor_directions = get_bottom_and_right_neighbor_directions(dist_th)
    # assuming elements of data1 are less than features_len
    angle_features = np.zeros(angle_f_len, dtype=np.float32)
    color_features = np.zeros(color_f_len, dtype=np.float32)
    for i in range(h):
        for j in range(w):
            center_color_enc = color_enc[i, j]
            center_angle_enc = angle_enc[i, j]
            for del_i, del_j in neighbor_directions:
                neighbor_i = i + del_i
                neighbor_j = j + del_j
                # ignore out of bounds neighbors and center pixel
                if (neighbor_i < 0 or neighbor_i >= h) or (neighbor_j < 0 or neighbor_j >= w):
                    continue
                neighbor_color_enc = color_enc[neighbor_i, neighbor_j]
                neighbor_angle_enc = angle_enc[neighbor_i, neighbor_j]

                if center_color_enc == neighbor_color_enc and center_angle_enc == neighbor_angle_enc:
                    color_diff = 0

                    # NOTE: assuming number of channels are 3
                    for channel in range(3):
                        color_diff += (img[i, j, channel] - img[neighbor_i, neighbor_j, channel]) ** 2
                    color_diff = np.sqrt(color_diff)
                    color_features[center_color_enc] += color_diff
                    angle_features[center_angle_enc] += color_diff

    features = np.concatenate([color_features, angle_features])
    return features


def get_feature_vector_from_img(img_rgb: np.ndarray) -> np.ndarray:
    pixels = (img_rgb.reshape(-1, 3).T).astype(np.float32) / 255
    lab_pixels = rgb_to_lab(pixels)

    img_lab = np.reshape(lab_pixels, img_rgb.shape)
    NB_COLORS = np.prod(LAB_NB_BINS)
    img_lab_quantized = quantize_lab(img_lab, LAB_NB_BINS)

    grad_direction_rad = compute_orientation(img_lab)
    grad_direction_quantized = quantize_angle(grad_direction_rad, NB_ANGLE_BINS)

    return extract_features_optimized(img_lab, img_lab_quantized, grad_direction_quantized, NB_COLORS, NB_ANGLE_BINS)
    # start1 = time.time()
    # features1 = extract_features1(img_lab, img_lab_quantized, grad_direction_quantized, NB_COLORS, NB_ANGLE_BINS)
    # end1 = time.time()
    #
    # start2 = time.time()
    # features2 = extract_features3(img_lab, img_lab_quantized, grad_direction_quantized, NB_COLORS, NB_ANGLE_BINS)
    # end2 = time.time()

    # print(np.abs(features1 - features2).sum())
    #
    # print(end1 - start1)
    # print(end2 - start2)
