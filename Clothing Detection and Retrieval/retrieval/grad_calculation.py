import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from configs import sobel_x, sobel_y


def calculate_grad(img: np.ndarray, in_x_direction: bool) -> np.ndarray:
    grad = np.zeros_like(img)
    filter = sobel_x if in_x_direction else sobel_y
    for i in range(img.shape[2]):
        grad[:, :, 0] = signal.correlate2d(img[:, :, 0], filter, 'same')
    return grad


def plot_roc_img(roc):
    max_roc = np.nanmax(roc, axis=0)
    min_roc = np.nanmin(roc, axis=0)

    max_roc = 255 * ((max_roc - np.min(max_roc)) / (np.max(max_roc) - np.min(max_roc)))
    min_roc = 255 * ((min_roc - np.min(min_roc)) / (np.max(min_roc) - np.min(min_roc)))

    plt.imshow(max_roc.astype(np.uint8), cmap='gray')
    plt.savefig('g1.png')
    plt.show()
    plt.imshow(min_roc.astype(np.uint8), cmap='gray')
    plt.savefig('g2.png')
    plt.show()


def compute_max_roc_direction(gxx, gyy, gxy):
    theta1_rad = .5 * np.arctan2(2 * gxy, gxx - gyy)
    theta2_rad = theta1_rad + (np.pi / 2)
    theta_combined = np.stack((theta1_rad, theta2_rad))
    roc = ((1 / 2) * (gxx + gyy +
                      ((gxx - gyy) * np.cos(2 * theta_combined)) +
                      (2 * gxy * np.sin(2 * theta_combined)))) ** (1 / 2)
    # plot_roc_img(roc)
    max_indices = np.argmax(roc, axis=0)
    mask = np.stack((1 - max_indices, max_indices))
    return (theta_combined * mask).sum(axis=0)


def compute_orientation(img: np.ndarray):
    img_x = calculate_grad(img, True)
    img_y = calculate_grad(img, False)
    gxx = (img_x ** 2).sum(2)
    gyy = (img_y ** 2).sum(2)
    gxy = (img_x * img_y).sum(2)
    return compute_max_roc_direction(gxx, gyy, gxy)


def quantize_angle(theta_rad: np.ndarray, bins: int = 18) -> np.ndarray:
    bin_size_rad = 2 * np.pi / bins

    # from -pi to pi -> 0 to 2 * pi
    theta_rad = np.where(theta_rad < 0, (2 * np.pi) + theta_rad, theta_rad)

    return np.floor(theta_rad / bin_size_rad).astype(np.int32)
