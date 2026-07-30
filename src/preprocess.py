import cv2
import numpy as np

MAX_LONGEST_SIDE = 1920


def preprocess_frame(frame):
    if frame is None or frame.size == 0:
        raise ValueError("Invalid frame")

    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    h, w = frame.shape[:2]
    if max(h, w) > MAX_LONGEST_SIDE:
        scale = MAX_LONGEST_SIDE / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return frame
