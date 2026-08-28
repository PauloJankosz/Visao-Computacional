"""Carrega as 32 imagens 512x512 em tons de cinza de data/dataset/."""

import os
import re
import cv2
FILENAME_PATTERN = re.compile(r"^(?:img_)?(\d+)\.png$")

class Dataset:
    def __init__(self, names, images):
        self.names = names
        self.images = images

def load_dataset(out_dir="data/dataset"):
    if not os.path.isdir(out_dir):
        return Dataset([], [])

    all_files = os.listdir(out_dir)

    names = []
    for file_name in all_files:
        if FILENAME_PATTERN.match(file_name):
            names.append(file_name)

    def file_number(name):
        match = FILENAME_PATTERN.match(name)
        return int(match.group(1))

    names.sort(key=file_number)

    images = []
    loaded_names = []
    for name in names:
        path = os.path.join(out_dir, name)
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            continue

        height, width = image.shape
        print(f"{name}: {width}x{height}, min={int(image.min())}, max={int(image.max())}")
        loaded_names.append(name)
        images.append(image)

    return Dataset(loaded_names, images)
