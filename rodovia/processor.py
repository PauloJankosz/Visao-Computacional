"""Pipeline único para vídeos gravados e transmissões HLS."""
import csv
import json
import time
from collections import Counter
from pathlib import Path
import cv2

from .calibration import Calibration
from .classify import classify_size
from .counting import LineCounter
from .detectors.classic import ClassicDetector
from .source import open_capture
from .speed import SpeedEstimator
from .tracker import CentroidTracker
from .visualize import draw


def process(source, method, output_dir='results', model='yolo11n.pt', line=None,
            direction='both', calibration_path=None, min_area=250, max_distance=70,
            conf=0.25, max_frames=None, device=None, no_video=False):
    if method not in ('classic','yolo'):
        raise ValueError('method deve ser classic ou yolo')
    cap = open_capture(source)
    writer = None
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(str(source)).stem if '://' not in str(source) else 'camera_ao_vivo'
    prefix = out / f'{stem}_{method}'
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not 1 <= fps <= 120:
        fps = 20.0
    live = '://' in str(source)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not width or not height:
        cap.release()
        raise RuntimeError('Não foi possível identificar a resolução do vídeo')
    if line is None:
        line = (0, height//2, width, height//2)
    counter = LineCounter(line, direction=direction)
    cal = Calibration.load(calibration_path) if calibration_path else None
    speed = SpeedEstimator(cal)
    if method == 'classic':
        detector = ClassicDetector(min_area=min_area)
        tracker = CentroidTracker(max_distance=max_distance)
    else:
        from .detectors.yolo import YoloVehicleTracker
        detector = YoloVehicleTracker(model=model, conf=conf, device=device)
        tracker = None
    count = 0
    started = time.perf_counter()
    output_video = str(prefix) + '.mp4'
    try:
        if not no_video:
            writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'),
                                     fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError('Codec mp4v indisponível para gravação')
        while max_frames is None or count < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            count += 1
            time_s = (time.perf_counter() - started) if live else (count-1)/fps
            detections = detector.detect(frame)
            if tracker:
                detections = tracker.update(detections, count)
                for item in detections:
                    item.label = classify_size(item.bbox)
            speed_by_id = {}
            for item in detections:
                # bottom center é aproximação do ponto de contato com o solo
                current_speed = speed.update(item.track_id, item.foot, time_s)
                speed_by_id[item.track_id] = current_speed
                counter.update(item, count, time_s, current_speed)
            if writer is not None:
                writer.write(draw(frame.copy(), detections, counter, speed_by_id))
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if not count:
        raise RuntimeError('Vídeo sem frames disponíveis')
    csv_file = str(prefix) + '_events.csv'
    with open(csv_file, 'w', newline='', encoding='utf-8') as handle:
        cols = ['frame','time_s','track_id','direction','label','speed_kmh','confidence']
        w = csv.DictWriter(handle, fieldnames=cols)
        w.writeheader()
        for event in counter.events:
            w.writerow(vars(event))
    summary = dict(source=str(source), method=method, frames=count, input_fps=fps,
                   processing_fps=count/max(time.perf_counter()-started,0.0001),
                   total=counter.total, by_label=dict(counter.counts),
                   line=list(line), direction=direction,
                   calibrated=cal is not None,
                   video=output_video if not no_video else None,
                   events_csv=csv_file,
                   caveat=('Velocidades apenas estimativas com homografia fornecida; '
                           'sem validação métrica não servem para fiscalização.'))
    json_file = str(prefix) + '_summary.json'
    Path(json_file).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    summary['summary_json'] = json_file
    return summary
