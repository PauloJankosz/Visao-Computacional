import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX

PALETTE = [    (255, 190, 90),
    (50, 50, 50),
    (150, 210, 130),
    (80, 160, 255),
    (210, 110, 255),
    (90, 220, 90),
    (255, 130, 130),
    (130, 255, 255),
    (255, 255, 120),
    (180, 180, 255),
]


def cluster_color(group):
    return PALETTE[group % len(PALETTE)]


def to_byte_image(values):
    norm = cv2.normalize(values, None, 0, 255, cv2.NORM_MINMAX)
    return norm.astype(np.uint8)


def gray_to_bgr(gray):
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def add_label(image, text, height=22):
    bar = np.full((height, image.shape[1], 3), 30, np.uint8)
    cv2.putText(bar, text, (4, height - 6), FONT, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    return np.vstack([bar, image])


def mosaic(images, cols=8, border=2, thumb=128):
    if not images:
        return np.zeros((thumb, thumb, 3), np.uint8)

    thumbs = []
    for img in images:
        if img.ndim == 2:
            img = gray_to_bgr(img)
        thumbs.append(cv2.resize(img, (thumb, thumb), interpolation=cv2.INTER_AREA))

    rows = []
    for start in range(0, len(thumbs), cols):
        row_imgs = thumbs[start : start + cols]
        while len(row_imgs) < cols:
            blank = np.full((thumb, thumb, 3), 20, np.uint8)
            row_imgs.append(blank)
        rows.append(cv2.hconcat(row_imgs))

    grid = cv2.vconcat(rows)
    if border > 0:
        grid = cv2.copyMakeBorder(grid, border, border, border, border, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    return grid


def colorize_labels(label_map, image_size):
    big = cv2.resize(label_map.astype(np.uint8), (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    color = np.zeros((image_size, image_size, 3), np.uint8)
    for group in range(int(label_map.max()) + 1):
        color[big == group] = cluster_color(group)
    return color


def overlay(gray, color, alpha=0.55):
    base = gray_to_bgr(gray)
    return cv2.addWeighted(base, 1.0 - alpha, color, alpha, 0)


def region_borders(label_map, image_size, thickness=2):
    big = cv2.resize(label_map.astype(np.uint8), (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    edges = np.zeros((image_size, image_size), np.uint8)
    for row in range(image_size):
        for col in range(1, image_size):
            if big[row, col] != big[row, col - 1]:
                edges[row, col] = 255
                edges[row, col - 1] = 255
    for row in range(1, image_size):
        for col in range(image_size):
            if big[row, col] != big[row - 1, col]:
                edges[row, col] = 255
                edges[row - 1, col] = 255
    if thickness > 1:
        kernel = np.ones((thickness, thickness), np.uint8)
        edges = cv2.dilate(edges, kernel)
    return edges


def legend(k, counts, width):
    height = 28
    bar = np.full((height, width, 3), 25, np.uint8)
    x = 8
    for group in range(k):
        color = cluster_color(group)
        cv2.rectangle(bar, (x, 8), (x + 14, 22), color, -1)
        text = f"G{group} ({int(counts[group])})"
        cv2.putText(bar, text, (x + 18, 20), FONT, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
        x += 90
    return bar


def result_panel(gray, label_map, title):
    color = colorize_labels(label_map, gray.shape[0])
    panel = overlay(gray, color)
    borders = region_borders(label_map, gray.shape[0], thickness=2)
    panel[borders > 0] = (255, 255, 255)
    return add_label(panel, title)


def filter_bank_figure(filters, n_scales=3, tile=64):
    rows = []
    for scale in range(n_scales):
        cols = []
        for filt in filters:
            kernel = filt.kernels[0]
            show = to_byte_image(kernel)
            if show.shape[0] != tile or show.shape[1] != tile:
                show = cv2.resize(show, (tile, tile), interpolation=cv2.INTER_NEAREST)
            cols.append(add_label(gray_to_bgr(show), f"e{scale} {filt.label[:12]}", height=20))
        rows.append(cv2.hconcat(cols))
    return cv2.vconcat(rows)


def response_grid(gray, maps, filters, n_scales=3, tile=120):
    rows = []
    for scale in range(n_scales):
        cols = []
        for filt in filters:
            key = f"e{scale}_{filt.name}"
            show = to_byte_image(maps[key])
            show = cv2.resize(show, (tile, tile), interpolation=cv2.INTER_AREA)
            cols.append(add_label(gray_to_bgr(show), f"e{scale} {filt.label[:12]}", height=20))
        rows.append(cv2.hconcat(cols))
    grid = cv2.vconcat(rows)
    input_img = cv2.resize(gray, (tile, tile), interpolation=cv2.INTER_AREA)
    side = np.full((grid.shape[0], tile + 8, 3), 20, np.uint8)
    top = (grid.shape[0] - tile) // 2
    left = 4
    side[top : top + tile, left : left + tile] = gray_to_bgr(input_img)
    return cv2.hconcat([side, grid])


def bar_chart(centers, labels):
    n_groups, n_dims = centers.shape
    width = max(640, n_dims * 28 + 80)
    height = 80 + n_groups * 36
    img = np.full((height, width, 3), 255, np.uint8)

    max_val = max(float(np.max(np.abs(centers))), 0.1)
    x0 = 70
    bar_w = (width - x0 - 20) // n_dims

    for group in range(n_groups):
        y = 40 + group * 36
        cv2.putText(img, f"G{group}", (8, y + 12), FONT, 0.45, (30, 30, 30), 1, cv2.LINE_AA)
        for dim in range(n_dims):
            val = centers[group, dim]
            x = x0 + dim * bar_w
            h = int(abs(val) / max_val * 24)
            color = (80, 80, 220) if val >= 0 else (220, 80, 80)
            cv2.rectangle(img, (x, y + 24 - h), (x + bar_w - 4, y + 24), color, -1)

    for dim in range(n_dims):
        x = x0 + dim * bar_w
        cv2.putText(img, labels[dim][-6:], (x, height - 8), FONT, 0.35, (60, 60, 60), 1, cv2.LINE_AA)

    return img


def image_groups_panel(images, names, image_labels, k_groups, thumb=128):
    blocks = []
    for group in range(k_groups):
        members = [i for i in range(len(images)) if image_labels[i] == group]
        if not members:
            continue
        thumbs = [cv2.resize(images[i], (thumb, thumb), interpolation=cv2.INTER_AREA) for i in members]
        row = mosaic([gray_to_bgr(t) for t in thumbs], cols=12, border=0, thumb=thumb)
        header = np.full((28, row.shape[1], 3), 32, np.uint8)
        cv2.rectangle(header, (8, 6), (24, 22), cluster_color(group), -1)
        cv2.putText(header, f"Class {group} - {len(members)} images", (32, 20), FONT, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
        blocks.append(np.vstack([header, row]))

    if not blocks:
        return np.zeros((thumb, thumb, 3), np.uint8)

    max_w = max(block.shape[1] for block in blocks)
    padded = []
    for block in blocks:
        if block.shape[1] < max_w:
            pad = max_w - block.shape[1]
            block = cv2.copyMakeBorder(block, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=(20, 20, 20))
        padded.append(block)
    return np.vstack(padded)
