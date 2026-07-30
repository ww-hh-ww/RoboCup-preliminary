from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps
from .base import FrameSource
from src.preprocess import preprocess_frame


class ImageFrameSource(FrameSource):

    def __init__(self, path):
        path = Path(path)
        pil_image = ImageOps.exif_transpose(Image.open(path))
        if pil_image is None:
            pil_image = Image.open(path)
        pil_image = pil_image.convert("RGB")
        self._frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        self._frame = preprocess_frame(self._frame)
        if self._frame is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        self._exhausted = False

    def read(self) -> np.ndarray | None:
        if self._exhausted:
            return None
        self._exhausted = True
        return self._frame.copy()
