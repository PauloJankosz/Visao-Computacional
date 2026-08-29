import csv
import sys
from collections import defaultdict

import numpy as np

INPUT_CSV = sys.argv[1] if len(sys.argv) > 1 else "results/vectors.csv"
OUTPUT_CSV = "results/cluster_table.csv"

COLUMNS = [
    ("e0_media", "brightness"),
    ("e0_gabor000", "0 deg"),
    ("e0_gabor045", "45 deg"),
    ("e0_gabor090", "90 deg"),
    ("e0_gabor135", "135 deg"),
    ("e0_log", "LoG"),
    ("e0_dog", "DoG"),
    ("e0_log2", "LoG coarse"),
    ("e2_gabor000", "0 deg (e2)"),
    ("e2_gabor090", "90 deg (e2)"),
]


def main():
    clusters = defaultdict(list)
    with open(INPUT_CSV, encoding="utf-8") as file:
        reader = csv.DictReader(file)
        dims = [name for name in reader.fieldnames if name.startswith("e")]
        for row in reader:
            clusters[int(row["cluster"])].append([float(row[name]) for name in dims])
    index = {name: i for i, name in enumerate(dims)}

    header = ["cluster", "regions", "%"] + [label for _, label in COLUMNS]
    rows = []
    total = sum(len(values) for values in clusters.values())
    for cluster_id in sorted(clusters):
        arr = np.array(clusters[cluster_id], dtype=np.float64)
        values = [arr[:, index[name]].mean() for name, _ in COLUMNS]
        rows.append([cluster_id, len(arr), 100.0 * len(arr) / total] + values)

    width = max(len(name) for name in header) + 2
    print("".join(name.rjust(width) for name in header))
    for row in rows:
        print(
            f"{row[0]:>{width}}{row[1]:>{width}}{row[2]:>{width}.1f}"
            + "".join(f"{value:>{width}.2f}" for value in row[3:])
        )

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for row in rows:
            writer.writerow([row[0], row[1], f"{row[2]:.1f}"] + [f"{value:.3f}" for value in row[3:]])
    print(f"\nSaved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
