import csv
import json
import tempfile
import unittest
from pathlib import Path

from rodovia.calibration import Calibration
from rodovia.classify import classify_size
from rodovia.counting import LineCounter
from rodovia.evaluate import evaluate_counts
from rodovia.models import Detection
from rodovia.source import hls_url
from rodovia.tracker import CentroidTracker
from rodovia.speed import SpeedEstimator


class TestGeometryAndCounting(unittest.TestCase):
    def test_line_direction_and_dedup(self):
        counter = LineCounter((0,50,100,50),direction='a_to_b')
        for i,y in enumerate([12,25,40,55,70,40,75]):
            d = Detection((20,y-10,45,y), 'carro',track_id=2)
            counter.update(d, i, i/30)
        self.assertEqual(counter.total,1)
        self.assertEqual(counter.events[0].direction,'a_to_b')

    def test_wrong_direction_is_rejected(self):
        counter = LineCounter((0,50,100,50),direction='b_to_a')
        for i,y in enumerate([10,20,35,65]):
            counter.update(Detection((10,y-10,30,y),track_id=3),i,i/30)
        self.assertEqual(counter.total,0)

    def test_centroid_tracking_retains_id(self):
        t=CentroidTracker(max_distance=30,max_missing=2)
        a=t.update([Detection((0,0,20,20))],1)[0]
        b=t.update([Detection((2,2,22,22))],2)[0]
        self.assertEqual(a.track_id,b.track_id)
        t.update([],3)
        c=t.update([Detection((4,4,24,24))],4)[0]
        self.assertEqual(c.track_id,a.track_id)

    def test_calibration_and_speed(self):
        cal=Calibration.from_correspondences([[0,0],[100,0],[100,100],[0,100]],
                                             [[0,0],[10,0],[10,10],[0,10]])
        self.assertAlmostEqual(cal.transform((50,50))[0],5,places=5)
        s=SpeedEstimator(cal,window_seconds=1)
        self.assertIsNone(s.update(3,(0,0),0))
        self.assertAlmostEqual(s.update(3,(100,0),1),36,places=3)

    def test_class_labels_are_sizes_only(self):
        self.assertEqual(classify_size((0,0,10,10)),'porte_pequeno')
        self.assertEqual(classify_size((0,0,70,70)),'porte_grande')

    def test_hls_id_validation(self):
        self.assertIn('cam_14',hls_url(14))
        with self.assertRaises(ValueError):
            hls_url('../14')

    def test_evaluation(self):
        with tempfile.TemporaryDirectory() as td:
            filename=Path(td)/'manual.csv'
            filename.write_text('video,total\nfile.mp4,5\n',encoding='utf-8')
            rows=evaluate_counts(filename,[{'source':'file.mp4','method':'classic','total':4}])
            self.assertEqual(rows[0]['absolute_error'],1)
            self.assertEqual(rows[0]['relative_error_pct'],20)


class TestIntegration(unittest.TestCase):
    def test_synthetic_classic_pipeline(self):
        from rodovia.demo import generate_demo
        from rodovia.processor import process
        with tempfile.TemporaryDirectory() as td:
            video=str(Path(td)/'demo.mp4')
            generate_demo(video)
            result=process(video,'classic',output_dir=td,line=[0,180,640,180],
                           min_area=180, no_video=True)
            self.assertEqual(result['frames'],160)
            self.assertTrue(Path(result['events_csv']).exists())
            self.assertTrue(Path(result['summary_json']).exists())
            self.assertGreaterEqual(result['total'],1)


if __name__ == '__main__':
    unittest.main()

class TestYoloAdapterWithoutWeights(unittest.TestCase):
    """Testa interface do adaptador YOLO com stub, NÃO avalia a rede neural."""
    def test_yolo_tracking_parsing(self):
        import sys
        from types import SimpleNamespace
        from unittest.mock import patch
        import numpy as np
        from rodovia.detectors.yolo import YoloVehicleTracker

        class Tensor:
            def __init__(self, arr): self.arr = np.asarray(arr)
            def cpu(self): return self
            def numpy(self): return self.arr
            def int(self): return Tensor(self.arr.astype(int))
            def tolist(self): return self.arr.tolist()
        fake_boxes = SimpleNamespace(
            is_track=True,
            id=Tensor([9]),
            cls=Tensor([2]),
            conf=Tensor([0.9]),
            xyxy=Tensor([[10, 20, 40, 60]]))
        class FakeYOLO:
            names={2:'car',3:'motorcycle',5:'bus',7:'truck'}
            def __init__(self, path): self.path=path
            def track(self, img, **kwargs):
                self.last_kwargs=kwargs
                return [SimpleNamespace(boxes=fake_boxes)]
        fake = SimpleNamespace(YOLO=FakeYOLO)
        with patch.dict(sys.modules, {'ultralytics':fake}):
            adapter=YoloVehicleTracker(model='fake.pt')
            detections=adapter.detect(np.zeros((80,80,3),dtype=np.uint8))
        self.assertEqual(len(detections),1)
        self.assertEqual(detections[0].label,'carro')
        self.assertEqual(detections[0].track_id,9)
        self.assertEqual(adapter.model.last_kwargs['tracker'],'bytetrack.yaml')
        self.assertTrue(adapter.model.last_kwargs['persist'])
