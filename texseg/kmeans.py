import numpy as np


def assign_clusters(data, centers):
    labels = np.zeros(len(data), dtype=np.int32)

    for i in range(len(data)):
        best_group = 0
        best_distance = 0.0

        diff = data[i] - centers[0]
        best_distance = float(np.dot(diff, diff))

        for group in range(1, len(centers)):
            diff = data[i] - centers[group]
            distance = float(np.dot(diff, diff))
            if distance < best_distance:
                best_distance = distance
                best_group = group

        labels[i] = best_group

    return labels


def compute_centers(data, labels, k):
    n_dims = data.shape[1]
    centers = np.zeros((k, n_dims), dtype=np.float64)

    for group in range(k):
        points = data[labels == group]
        if len(points) == 0:
            random_index = np.random.randint(0, len(data))
            centers[group] = data[random_index]
        else:
            centers[group] = points.mean(axis=0)

    return centers


def total_error(data, labels, centers):
    error = 0.0
    for i in range(len(data)):
        diff = data[i] - centers[labels[i]]
        error += float(np.dot(diff, diff))
    return error


def kmeans_once(data, k, max_iters=100):
    n_points = len(data)
    start_indices = np.random.choice(n_points, k, replace=False)
    centers = data[start_indices].copy()

    for _ in range(max_iters):
        labels = assign_clusters(data, centers)
        new_centers = compute_centers(data, labels, k)

        if np.allclose(new_centers, centers):
            break

        centers = new_centers

    labels = assign_clusters(data, centers)
    error = total_error(data, labels, centers)
    return labels, centers, error


def kmeans(data, k, n_init=8):
    best_labels = None
    best_centers = None
    best_error = float("inf")

    for _ in range(n_init):
        labels, centers, error = kmeans_once(data, k)
        if error < best_error:
            best_error = error
            best_labels = labels
            best_centers = centers

    return best_labels, best_centers, best_error


def sort_clusters(labels, centers, scores):
    order = np.argsort(scores)
    mapping = np.zeros(len(centers), dtype=np.int32)

    for new_id, old_id in enumerate(order):
        mapping[old_id] = new_id

    new_labels = mapping[labels]
    new_centers = centers[order]
    return new_labels, new_centers
