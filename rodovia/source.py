"""Vídeos gravados, URLs HLS e consulta opcional a catálogo de câmeras."""
import json
from urllib.request import urlopen
from urllib.error import URLError
import cv2

DER_BASE = 'http://200.144.30.103:8084'


def list_cameras(url=DER_BASE+'/api/cameras', timeout=8):
    try:
        with urlopen(url, timeout=timeout) as response:
            return json.load(response)
    except (URLError, TimeoutError, ValueError, OSError) as exc:
        raise RuntimeError(f'Catálogo indisponível: {exc}') from exc


def hls_url(camera_id, base=DER_BASE):
    if not str(camera_id).isdigit():
        raise ValueError('ID de câmera deve ser numérico')
    return f'{base}/hls/cam_{camera_id}/stream.m3u8'


def open_capture(source):
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f'Não foi possível abrir a fonte: {source}')
    return cap
