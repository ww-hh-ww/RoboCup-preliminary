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
        if not self.gallery.embeddings:
            raise RuntimeError(
                f"Face gallery is empty at {gallery_path}. "
                "Build it first with: python -m src.face.build_gallery ..."
            )
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
                "top1_name": match["top1_name"],
                "top1_similarity": match["confidence"],
                "top2_name": match["top2_name"],
                "top2_similarity": match["top2_similarity"],
                "match_status": match["match_status"],
            })
        return results
