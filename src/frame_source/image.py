from pathlib import Path
import cv2
import numpy as np
from .base import FrameSource


class ImageFrameSource(FrameSource):

    def __init__(self, path):
        self._frame = cv2.imread(str(Path(path)))
        if self._frame is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        self._exhausted = False

    def read(self) -> np.ndarray | None:
        if self._exhausted:
            return None
        self._exhausted = True
        return self._frame.copy()
