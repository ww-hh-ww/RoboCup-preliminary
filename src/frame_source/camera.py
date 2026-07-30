import cv2
import numpy as np
from .base import FrameSource


class CameraFrameSource(FrameSource):

    def __init__(self, camera_id=0):
        self._cap = cv2.VideoCapture(camera_id)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")

    def read(self) -> np.ndarray | None:
        ret, frame = self._cap.read()
        return frame if ret else None

    def release(self):
        self._cap.release()
