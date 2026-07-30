from pathlib import Path
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from .build_gallery import FaceGallery


class FaceDetector:

    def __init__(self, gallery_path="data/processed/face_gallery", threshold=0.4, det_size=(640, 640)):
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
            root=Path.home() / ".insightface",
        )
        self.app.prepare(ctx_id=0, det_size=det_size)
        self.gallery = FaceGallery(gallery_path)
        self.threshold = threshold

    def detect(self, frame):
        faces = self.app.get(frame)
        results = []
        for face in faces:
            emb = face.normed_embedding
            match = self.gallery.match(emb, self.threshold)
            bbox = face.bbox.astype(int).tolist()
            results.append({
                "bbox": bbox,
                "name": match["name"],
                "gender": match["gender"],
                "confidence": match["confidence"],
                "face_confidence": float(face.det_score),
            })
        return results
