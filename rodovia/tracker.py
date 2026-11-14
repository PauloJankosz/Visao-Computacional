"""Rastreador simples de centroide: comparação com ByteTrack, não equivalente a ele."""
import math
from .models import Track


class CentroidTracker:
    def __init__(self, max_distance=70, max_missing=12):
        self.max_distance = max_distance
        self.max_missing = max_missing
        self.tracks = {}
        self.next_id = 1

    def update(self, detections, frame_index):
        # Associação global gulosa pela distância. Somente os tracks observados
        # neste frame são retornados (tracks perdidos não contam/passam na linha).
        candidates = []
        for tid, track in self.tracks.items():
            if track.missing > self.max_missing:
                continue
            cx, cy = track.detection.center
            for j, det in enumerate(detections):
                dx, dy = det.center
                distance = math.hypot(cx-dx, cy-dy)
                if distance <= self.max_distance:
                    candidates.append((distance, tid, j))
        matched_ids, matched_det = set(), set()
        for _, tid, j in sorted(candidates):
            if tid in matched_ids or j in matched_det:
                continue
            matched_ids.add(tid)
            matched_det.add(j)
            t = self.tracks[tid]
            t.detection = detections[j]
            t.last_frame = frame_index
            t.hits += 1
            t.missing = 0
            t.path.append(detections[j].foot)
            t.path = t.path[-64:]
            detections[j].track_id = tid
        for tid in list(self.tracks):
            if tid not in matched_ids:
                self.tracks[tid].missing += 1
                if self.tracks[tid].missing > self.max_missing:
                    del self.tracks[tid]
        for j, det in enumerate(detections):
            if j in matched_det:
                continue
            tid = self.next_id
            self.next_id += 1
            det.track_id = tid
            self.tracks[tid] = Track(tid, det, frame_index, frame_index, path=[det.foot])
        return detections
