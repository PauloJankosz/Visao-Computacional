import math
import cv2
import numpy as np

class Filter:
    def __init__(self, name, label, kernels, mode):
        self.name = name
        self.label = label
        self.kernels = kernels
        self.mode = mode


def normalize_kernel(kernel):
    k = kernel.astype(np.float32)
    k = k - k.mean()
    total = np.abs(k).sum()
    if total > 0:
        k = k / total
    return k


def make_grid(size):
    center = (size - 1) / 2.0
    ys = np.arange(size) - center
    xs = np.arange(size) - center
    xx, yy = np.meshgrid(xs, ys)
    return xx, yy


def gabor_kernel(angle_deg, psi=0.0, size=15):
    sigma = 2.4
    lambd = 5.0
    gamma = 0.6

    theta = math.radians(angle_deg + 90.0)
    xx, yy = make_grid(size)
    x_rot = xx * math.cos(theta) + yy * math.sin(theta)
    y_rot = -xx * math.sin(theta) + yy * math.cos(theta)

    gauss = np.exp(-(x_rot ** 2 + gamma ** 2 * y_rot ** 2) / (2 * sigma ** 2))
    wave = np.cos(2 * math.pi * x_rot / lambd + psi)
    kernel = gauss * wave
    return normalize_kernel(kernel)


def log_kernel(size=15, sigma=1.4):
    xx, yy = make_grid(size)
    r2 = xx ** 2 + yy ** 2
    s2 = sigma ** 2
    kernel = (r2 - 2 * s2) / (s2 ** 2) * np.exp(-r2 / (2 * s2))
    return normalize_kernel(kernel)


def dog_kernel(size=15, sigma1=2.0, sigma2=3.2):
    xx, yy = make_grid(size)
    r2 = xx ** 2 + yy ** 2
    g_small = np.exp(-r2 / (2 * sigma1 ** 2))
    g_large = np.exp(-r2 / (2 * sigma2 ** 2))
    kernel = g_small - g_large
    return normalize_kernel(kernel)


def gaussian_mean_kernel(size=15, sigma=2.0):
    g1 = cv2.getGaussianKernel(size, sigma)
    kernel = g1 @ g1.T
    total = kernel.sum()
    if total > 0:
        kernel = kernel / total
    return kernel.astype(np.float32)


def build_filter_bank():
    filters = []

    for angle in (0, 45, 90, 135):
        even = gabor_kernel(angle, psi=0.0)
        odd = gabor_kernel(angle, psi=math.pi / 2)
        filters.append(
            Filter(
                name=f"gabor{angle:03d}",
                label=f"Gabor {angle} deg",
                kernels=[even, odd],
                mode="energy",
            )
        )

    filters.append(
        Filter(
            name="log",
            label="LoG",
            kernels=[log_kernel()],
            mode="abs",
        )
    )

    filters.append(
        Filter(
            name="dog",
            label="DoG",
            kernels=[dog_kernel()],
            mode="abs",
        )
    )

    filters.append(
        Filter(
            name="log2",
            label="LoG coarse",
            kernels=[log_kernel(sigma=2.8)],
            mode="abs",
        )
    )

    filters.append(
        Filter(
            name="media",
            label="Local mean",
            kernels=[gaussian_mean_kernel()],
            mode="linear",
        )
    )

    return filters
