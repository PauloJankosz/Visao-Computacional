"""Homografia imagem->plano da pista em metros, com pontos conhecidos."""
import json
from pathlib import Path
import cv2
import numpy as np


class Calibration:
    def __init__(self, matrix):
        self.matrix = np.asarray(matrix, dtype=np.float64)
        if self.matrix.shape != (3, 3) or not np.isfinite(self.matrix).all():
            raise ValueError('Matriz de homografia inválida')

    @classmethod
    def from_correspondences(cls, pixel_points, meter_points):
        if len(pixel_points) != len(meter_points) or len(pixel_points) < 4:
            raise ValueError('Mínimo de quatro correspondências pixels->metros')
        src = np.asarray(pixel_points, dtype=np.float32)
        dst = np.asarray(meter_points, dtype=np.float32)
        h, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 3)
        if h is None or abs(np.linalg.det(h)) < 1e-12:
            raise ValueError('Não foi possível calibrar; confira pontos não colineares')
        return cls(h)

    def transform(self, point):
        p = np.array([[[point[0], point[1]]]], dtype=np.float64)
        result = cv2.perspectiveTransform(p, self.matrix)
        return float(result[0, 0, 0]), float(result[0, 0, 1])

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        if 'homography' in data:
            return cls(data['homography'])
        return cls.from_correspondences(data['image_points'], data['world_points_m'])

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({'homography':self.matrix.tolist()}, indent=2), encoding='utf-8')
