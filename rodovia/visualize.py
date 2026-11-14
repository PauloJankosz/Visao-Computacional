"""Sobreposição visual nos vídeos gerados."""
import cv2


def draw(frame, detections, counter, speeds=None):
    speeds = speeds or {}
    p1 = tuple(int(round(v)) for v in counter.line[:2])
    p2 = tuple(int(round(v)) for v in counter.line[2:])
    cv2.line(frame, p1, p2, (0, 210, 255), 2)
    for d in detections:
        x1,y1,x2,y2 = d.bbox
        cv2.rectangle(frame, (x1,y1), (x2,y2), (20,230,70), 2)
        parts = [f'#{d.track_id}', d.label]
        if d.track_id in speeds and speeds[d.track_id] is not None:
            parts.append(f'{speeds[d.track_id]:.1f}km/h')
        text = ' '.join(parts)
        cv2.putText(frame, text, (x1,max(16,y1-5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (255,255,255), 2)
        cv2.putText(frame, text, (x1,max(16,y1-5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (10,80,10), 1)
    cv2.rectangle(frame, (0,0), (280,55), (0,0,0), -1)
    cv2.putText(frame, f'Contagem: {counter.total}', (10,22), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255,255,255), 2)
    cv2.putText(frame, 'Linha amarela = contagem', (10,44), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (220,220,220), 1)
    return frame
