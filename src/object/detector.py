from pathlib import Path
from ultralytics import YOLO


class ObjectDetector:

    def __init__(self, model_path="models/object/best.pt", fallback="yolo11n.pt", strict=False):
        path = Path(model_path)
        if strict and not path.exists():
            raise FileNotFoundError(
                f"Custom model not found: {path}. "
                "Train a model first or place the weights file at the expected path."
            )
        self.model = YOLO(str(path) if path.exists() else fallback)
        self.class_names = self.model.names

    def detect(self, frame, conf_threshold=0.25):
        results = self.model(frame, conf=conf_threshold, verbose=False)[0]
        detections = []
        for det in results.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls_id = det
            cls_id = int(cls_id)
            class_name = self.class_names.get(cls_id, f"class_{cls_id}")
            detections.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "class_id": cls_id,
                "class_name": class_name,
                "confidence": float(conf),
            })
        return detections
