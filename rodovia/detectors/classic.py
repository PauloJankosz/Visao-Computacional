"""Baseline clássico: MOG2 + operações morfológicas + contornos."""
import cv2
from ..models import Detection


class ClassicDetector:
    def __init__(self, min_area=250, max_area=300000, warmup=20,
                 history=450, var_threshold=32, shadows=False):
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=shadows)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.min_area = min_area
        self.max_area = max_area
        self.warmup = warmup
        self.frames_seen = 0

    def detect(self, frame):
        self.frames_seen += 1
        mask = self.bg.apply(frame)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        if self.frames_seen <= self.warmup:
            return []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if not self.min_area <= area <= self.max_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w < 9 or h < 9:
                continue
            found.append(Detection((x, y, x + w, y + h)))
        return found
