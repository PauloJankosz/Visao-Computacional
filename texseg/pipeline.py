import csv
import json
import os
import time

import cv2
import numpy as np

from . import visualize as vis
from .dataset import load_dataset
from .features import N_SCALES, WINDOW, dimension_names, extract_features, normalize_features
from .filters import build_filter_bank
from .kmeans import kmeans, sort_clusters

BRIGHTNESS_DIMS = [7, 15, 23]


def save_image(path, image):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, 92])


def run(
    dataset_dir="data/dataset",
    out_dir="results",
    k=6,
    window=WINDOW,
    k_images=4,
):
    start = time.time()
    os.makedirs(out_dir, exist_ok=True)

    print("[1/6] loading dataset")
    dataset = load_dataset(dataset_dir)
    if len(dataset.images) == 0:
        raise SystemExit(f"No images found in '{dataset_dir}'.")

    grid_size = dataset.images[0].shape[0] // window
    filters = build_filter_bank()
    dim_names = dimension_names(filters)
    short_labels = [f.label[:10] for _ in range(N_SCALES) for f in filters]

    thumbs = [cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA) for img in dataset.images]
    save_image(os.path.join(out_dir, "fig00_dataset_sample.png"), vis.mosaic(thumbs, cols=8))

    print("[2/6] saving filter bank figure")
    save_image(os.path.join(out_dir, "fig01_filter_bank.png"), vis.filter_bank_figure(filters, N_SCALES))

    print("[3/6] extracting texture vectors")
    all_vectors = []
    first_maps = None
    for i, image in enumerate(dataset.images):
        if i == 0:
            vectors, first_maps = extract_features(image, filters, window, with_maps=True)
        else:
            vectors = extract_features(image, filters, window)
        all_vectors.append(vectors)

    stacked = np.stack(all_vectors, axis=0)
    n_images = stacked.shape[0]
    n_dims = stacked.shape[-1]

    save_image(
        os.path.join(out_dir, "fig02_filter_responses.png"),
        vis.response_grid(dataset.images[0], first_maps, filters, N_SCALES),
    )

    flat, _, _ = normalize_features(stacked)
    print(f"      {flat.shape[0]} regions x {n_dims} dimensions")

    print(f"[4/6] k-means with K={k}")
    labels, centers, sse = kmeans(flat, k, n_init=8)

    texture_dims = [i for i in range(n_dims) if i not in BRIGHTNESS_DIMS]
    scores = centers[:, texture_dims].mean(axis=1)
    labels, centers = sort_clusters(labels, centers, scores)

    counts = np.bincount(labels, minlength=k)
    save_image(
        os.path.join(out_dir, "fig03_cluster_profiles.png"),
        vis.bar_chart(centers, short_labels),
    )

    print("[5/6] segmentation maps")
    label_maps = labels.reshape(n_images, grid_size, grid_size)
    seg_dir = os.path.join(out_dir, "segmentation")
    overlays = []

    for i in range(n_images):
        name = dataset.names[i]
        image = dataset.images[i]
        label_map = label_maps[i]

        panel = vis.result_panel(image, label_map, name)
        legend = vis.legend(k, np.bincount(label_map.ravel(), minlength=k), panel.shape[1])
        out_name = os.path.splitext(name)[0] + ".jpg"
        save_image(os.path.join(seg_dir, out_name), np.vstack([panel, legend]))

        color = vis.colorize_labels(label_map, image.shape[0])
        blend = vis.overlay(image, color)
        borders = vis.region_borders(label_map, image.shape[0], thickness=2)
        blend[borders > 0] = (255, 255, 255)
        overlays.append(cv2.resize(blend, (192, 192), interpolation=cv2.INTER_AREA))

    mosaic = vis.mosaic(overlays, cols=8)
    mosaic = np.vstack([mosaic, vis.legend(k, counts, mosaic.shape[1])])
    save_image(os.path.join(out_dir, "fig04_segmentation_mosaic.png"), mosaic)

    print(f"[6/6] grouping {n_images} images into {k_images} classes")
    image_profiles = []
    for i in range(n_images):
        mean_vector = stacked[i].reshape(-1, n_dims).mean(axis=0)
        image_profiles.append(mean_vector)
    image_profiles = np.stack(image_profiles, axis=0)
    profiles_norm, _, _ = normalize_features(image_profiles)

    image_labels, image_centers, _ = kmeans(profiles_norm, k_images, n_init=10)
    image_labels, image_centers = sort_clusters(
        image_labels, image_centers, image_centers.mean(axis=1)
    )

    save_image(
        os.path.join(out_dir, "fig05_image_groups.png"),
        vis.image_groups_panel(dataset.images, dataset.names, image_labels, k_images),
    )

    print("      saving CSV and JSON")
    with open(os.path.join(out_dir, "vectors.csv"), "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image", "row", "col", "cluster", *dim_names])
        for i in range(n_images):
            for row in range(grid_size):
                for col in range(grid_size):
                    group = int(label_maps[i, row, col])
                    values = stacked[i, row, col]
                    writer.writerow([dataset.names[i], row, col, group, *[f"{v:.5f}" for v in values]])

    with open(os.path.join(out_dir, "centers.csv"), "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["cluster", "regions", *dim_names])
        for group in range(k):
            writer.writerow([group, int(counts[group]), *[f"{v:.4f}" for v in centers[group]]])

    summary = {
        "n_images": int(n_images),
        "dimensions": int(n_dims),
        "scales": int(N_SCALES),
        "window": int(window),
        "grid_per_image": f"{grid_size}x{grid_size}",
        "total_regions": int(flat.shape[0]),
        "k_regions": int(k),
        "sse": round(float(sse), 2),
        "cluster_counts": {f"G{group}": int(counts[group]) for group in range(k)},
        "k_images": int(k_images),
        "image_classes": {dataset.names[i]: int(image_labels[i]) for i in range(n_images)},
        "elapsed_s": round(time.time() - start, 1),
    }

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(f"\nDone in {summary['elapsed_s']}s. Results saved to '{out_dir}'.")
    return summary
