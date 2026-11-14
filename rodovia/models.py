"""Estruturas comuns aos dois pipelines de análise de veículos."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]  # x1,y1,x2,y2
    label: str = 'veiculo'
    confidence: Optional[float] = None
    track_id: Optional[int] = None

    @property
    def foot(self) -> tuple[float, float]:
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, float(y2))

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass
class CrossingEvent:
    frame: int
    time_s: float
    track_id: int
    direction: str
    label: str
    speed_kmh: Optional[float] = None
    confidence: Optional[float] = None


@dataclass
class Track:
    id: int
    detection: Detection
    last_frame: int
    first_frame: int
    hits: int = 1
    missing: int = 0
    path: list[tuple[float, float]] = field(default_factory=list)
