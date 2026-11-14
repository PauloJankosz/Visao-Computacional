"""Contagem de cruzamentos de linha por identificador de rastreamento."""
from collections import Counter
from .models import CrossingEvent


class LineCounter:
    def __init__(self, line, direction='both', margin=2.0, min_hits=2):
        if direction not in ('both','a_to_b','b_to_a'):
            raise ValueError('Direção inválida')
        x1,y1,x2,y2 = line
        if x1 == x2 and y1 == y2:
            raise ValueError('Linha precisa de dois pontos distintos')
        self.line = tuple(float(v) for v in line)
        self.direction = direction
        self.margin = margin
        self.min_hits = min_hits
        self.sides = {}
        self.seen = set()
        self.hit_counts = Counter()
        self.counts = Counter()
        self.events = []

    def _side(self, point):
        x1,y1,x2,y2 = self.line
        x,y = point
        cross = (x2-x1)*(y-y1) - (y2-y1)*(x-x1)
        length = ((x2-x1)**2 + (y2-y1)**2)**0.5
        signed_dist = cross / length
        return 1 if signed_dist > self.margin else -1 if signed_dist < -self.margin else 0

    def update(self, detection, frame, time_s, speed_kmh=None):
        tid = detection.track_id
        if tid is None:
            return None
        self.hit_counts[tid] += 1
        side = self._side(detection.foot)
        if side == 0:
            return None
        old = self.sides.get(tid)
        self.sides[tid] = side
        if old is None or old == side or tid in self.seen:
            return None
        if self.hit_counts[tid] < self.min_hits:
            return None
        direction = 'a_to_b' if old < side else 'b_to_a'
        if self.direction not in ('both', direction):
            return None
        self.seen.add(tid)
        self.counts[detection.label] += 1
        event = CrossingEvent(frame=frame, time_s=time_s, track_id=tid,
                              direction=direction, label=detection.label,
                              speed_kmh=speed_kmh, confidence=detection.confidence)
        self.events.append(event)
        return event

    @property
    def total(self):
        return sum(self.counts.values())
