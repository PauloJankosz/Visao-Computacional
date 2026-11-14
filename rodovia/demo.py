"""Gera vídeo sintético com verdade conhecida: TESTA pipeline, não acurácia real."""
from pathlib import Path
import cv2
import numpy as np


def generate_demo(path, total_frames=160, fps=20, width=640, height=360):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError('OpenCV não conseguiu escrever mp4v')
    try:
        for frame_idx in range(total_frames):
            img = np.full((height,width,3), (70,70,70), dtype=np.uint8)
            cv2.rectangle(img,(40,0),(600,height),(76,76,76),-1)
            cv2.line(img,(0,180),(width,180),(128,128,128),1)
            # 3 objetos sempre presentes em faixas distintas, sem oclusão
            for x, y0, speed, color in [
                (120,-90,2.2,(0,180,240)),
                (300,-120,2.7,(200,150,20)),
                (470,-145,2.5,(150,100,230)),
            ]:
                y = int(y0 + speed*frame_idx)
                if -32 < y < height:
                    cv2.rectangle(img,(x-24,y),(x+24,y+28),color,-1)
            writer.write(img)
    finally:
        writer.release()
    return {'video':str(path),'ground_truth':3,'counting_line':[0,180,width,180]}
