"""Estimativa demonstrativa; km/h SOMENTE com calibração métrica confiável."""
from collections import deque
import math


class SpeedEstimator:
    def __init__(self, calibration=None, window_seconds=0.7, stale_s=10):
        self.calibration = calibration
        self.window_seconds = window_seconds
        self.stale_s = stale_s
        self.history = {}

    def update(self, track_id, foot, time_s):
        if self.calibration is None:
            return None
        xy = self.calibration.transform(foot)
        h = self.history.setdefault(track_id, deque())
        if h and time_s <= h[-1][0]:
            return None
        h.append((time_s, xy))
        # Manter amostras suficientes para estimar em intervalo significativo.
        while len(h) > 2 and (time_s - h[1][0] >= self.window_seconds):
            h.popleft()
        if len(h) < 2 or time_s - h[0][0] < 0.15:
            return None
        elapsed = time_s - h[0][0]
        return 3.6 * math.dist(xy, h[0][1]) / elapsed
