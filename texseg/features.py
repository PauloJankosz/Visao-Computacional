import cv2
import numpy as np
from .filters import build_filter_bank
N_SCALES = 3
WINDOW = 32


def build_pyramid(image, n_levels=N_SCALES):
    levels = [image.astype(np.float32)]

    for _ in range(n_levels - 1):
        smooth = cv2.GaussianBlur(levels[-1], (5, 5), 1.0)
        smaller = smooth[::2, ::2]
        levels.append(smaller)

    return levels


def apply_filter(image, filt):
    responses = []
    for kernel in filt.kernels:
        resp = cv2.filter2D(image, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT_101)
        responses.append(resp)

    if filt.mode == "energy":
        return np.sqrt(responses[0] ** 2 + responses[1] ** 2)

    if filt.mode == "magnitude":
        return np.sqrt(responses[0] ** 2 + responses[1] ** 2)

    if filt.mode == "abs":
        return np.abs(responses[0])

    return responses[0]


def mean_per_window(response_map, grid_size):
    height, width = response_map.shape
    cell_h = height // grid_size
    cell_w = width // grid_size

    result = np.zeros((grid_size, grid_size), dtype=np.float32)

    for row in range(grid_size):
        for col in range(grid_size):
            y1 = row * cell_h
            y2 = y1 + cell_h
            x1 = col * cell_w
            x2 = x1 + cell_w
            block = response_map[y1:y2, x1:x2]
            result[row, col] = block.mean()

    return result


def dimension_names(filters=None):
    if filters is None:
        filters = build_filter_bank()

    names = []
    for scale in range(N_SCALES):
        for filt in filters:
            names.append(f"e{scale}_{filt.name}")
    return names


def extract_features(image, filters=None, window=WINDOW):
    if filters is None:
        filters = build_filter_bank()

    grid_size = image.shape[0] // window
    pyramid = build_pyramid(image)

    channels = []
    for level_image in pyramid:
        for filt in filters:
            response = apply_filter(level_image, filt)
            window_means = mean_per_window(response, grid_size)
            channels.append(window_means)

    vectors = np.stack(channels, axis=-1)
    return vectors


def normalize_features(vectors):
    flat = vectors.reshape(-1, vectors.shape[-1]).astype(np.float64)
    flat = np.log1p(np.maximum(flat, 0.0))

    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    for i in range(len(std)):
        if std[i] < 1e-8:
            std[i] = 1.0

    normalized = (flat - mean) / std
    return normalized, mean, std
