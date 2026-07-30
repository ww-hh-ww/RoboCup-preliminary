from pathlib import Path
from ultralytics import YOLO


class ObjectDetector:

    def __init__(self, model_path="models/object/best.pt", fallback="yolo11n.pt", strict=False):
        path = Path(model_path)
        local_fallback = Path("models") / fallback
        if strict and not path.exists():
            raise FileNotFoundError(
                f"Custom model not found: {path}. "
                "Train a model first or place the weights file at the expected path."
            )
        if path.exists():
            model_to_load = str(path)
        elif local_fallback.exists():
            model_to_load = str(local_fallback)
        else:
            model_to_load = fallback
        self.model = YOLO(model_to_load)
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
