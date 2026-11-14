"""Grava trechos de uma câmera/URL para experimentos reproduzíveis."""
import time
from pathlib import Path
import cv2
from .source import open_capture


def record(source, dest, seconds=20, fps_fallback=20.0):
    if seconds <= 0:
        raise ValueError('seconds deve ser positivo')
    cap = open_capture(source)
    writer = None
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    n = 0
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not 1 <= fps <= 120:
            fps = fps_fallback
        while time.monotonic() - t0 < seconds:
            ok, frame = cap.read()
            if not ok:
                break
            if writer is None:
                h,w = frame.shape[:2]
                writer = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*'mp4v'),
                                         fps, (w,h))
                if not writer.isOpened():
                    raise RuntimeError('Falha ao criar vídeo de saída (codec mp4v)')
            writer.write(frame)
            n += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if n == 0:
        raise RuntimeError('Nenhum frame gravado')
    return n
