import platform

import cv2
import numpy as np
from .base import FrameSource
from src.preprocess import preprocess_frame


class CameraFrameSource(FrameSource):

    def __init__(self, camera_id=0):
        if platform.system() == "Windows":
            self._cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        else:
            self._cap = cv2.VideoCapture(camera_id)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera resolution: {width}x{height}")
        for _ in range(20):
            self._cap.read()

    def read(self) -> np.ndarray | None:
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return None
        return preprocess_frame(frame)

    def release(self):
        self._cap.release()
