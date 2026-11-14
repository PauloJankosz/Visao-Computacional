"""Detector YOLO v11 + ByteTrack; import tardio para permitir baseline sem torch."""
from ..models import Detection

COCO_VEHICLES = {2: 'carro', 3: 'moto', 5: 'onibus', 7: 'caminhao'}


class YoloVehicleTracker:
    def __init__(self, model='yolo11n.pt', conf=0.25, device=None):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError('Ultralytics não instalado. Rode pip install -r requirements.txt') from exc
        self.model = YOLO(model)
        self.conf = conf
        self.device = device
        # O mapeamento de classes assume pesos COCO; conferir caso modelo custom.
        for idx, name in ((2,'car'), (3,'motorcycle'), (5,'bus'), (7,'truck')):
            model_name = str(self.model.names.get(idx, '')).lower()
            if model_name != name:
                raise ValueError('Pesos YOLO devem ter classes COCO. '
                                 f'Esperado {name} no id {idx}, recebido {model_name!r}')

    def detect(self, frame):
        kwargs = dict(persist=True, tracker='bytetrack.yaml', conf=self.conf,
                      classes=list(COCO_VEHICLES), verbose=False)
        if self.device:
            kwargs['device'] = self.device
        result = self.model.track(frame, **kwargs)[0]
        boxes = result.boxes
        if boxes is None or not boxes.is_track or boxes.id is None:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.int().cpu().tolist()
        confs = boxes.conf.cpu().tolist()
        ids = boxes.id.int().cpu().tolist()
        return [Detection(tuple(int(v) for v in box), COCO_VEHICLES[int(c)],
                          float(score), int(tid))
                for box, c, score, tid in zip(xyxy, cls, confs, ids)]
