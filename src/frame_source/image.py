from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ExifTags
from .base import FrameSource
from src.preprocess import preprocess_frame


class ImageFrameSource(FrameSource):

    def __init__(self, path):
        path = Path(path)
        pil_image = Image.open(path)
        exif = pil_image.getexif()
        orientation = exif.get(ExifTags.Base.Orientation)
        if orientation is not None and orientation != 1:
            rot_map = {
                3: 180,
                6: 90,
                8: 270,
            }
            if orientation in rot_map:
                pil_image = pil_image.rotate(rot_map[orientation], expand=True)
            elif orientation == 2:
                pil_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
            elif orientation == 4:
                pil_image = pil_image.transpose(Image.FLIP_TOP_BOTTOM)
            elif orientation == 5:
                pil_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT).rotate(270, expand=True)
            elif orientation == 7:
                pil_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT).rotate(90, expand=True)
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
