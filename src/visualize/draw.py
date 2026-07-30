from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


FACE_COLOR = (0, 255, 0)
OBJECT_COLOR = (255, 0, 0)
THICKNESS = 2

_FONT_CACHE = None


def _get_font(size=16):
    global _FONT_CACHE
    if _FONT_CACHE is not None:
        return _FONT_CACHE
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for p in candidates:
        if Path(p).exists():
            _FONT_CACHE = ImageFont.truetype(p, size)
            return _FONT_CACHE
    _FONT_CACHE = ImageFont.load_default()
    return _FONT_CACHE


def _to_pil(img_bgr):
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def _to_cv2(img_pil):
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def _draw_label(draw, text, pos, color, font):
    x, y = pos
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    if y < th + 6:
        y = th + 6
    r, g, b = color
    draw.rectangle([x, y - th - 6, x + tw + 4, y + 4], fill=(r, g, b))
    draw.text((x + 2, y - th - 4), text, font=font, fill=(255, 255, 255))


def draw_results(frame, face_results, object_results):
    pil = _to_pil(frame)
    draw = ImageDraw.Draw(pil)
    font = _get_font()

    for obj in object_results:
        x1, y1, x2, y2 = obj["bbox"]
        label = f"{obj['class_name']} {obj['confidence']:.2f}"
        r, g, b = OBJECT_COLOR
        draw.rectangle([x1, y1, x2, y2], outline=(r, g, b), width=THICKNESS)
        _draw_label(draw, label, (x1, y1), OBJECT_COLOR, font)

    for face in face_results:
        x1, y1, x2, y2 = face["bbox"]
        gender = face.get("gender") or ""
        name = face["name"]
        label = f"{name} {gender}".strip()
        r, g, b = FACE_COLOR
        draw.rectangle([x1, y1, x2, y2], outline=(r, g, b), width=THICKNESS)
        _draw_label(draw, label, (x1, y1), FACE_COLOR, font)

    return _to_cv2(pil)
