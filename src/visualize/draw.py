import cv2


FACE_COLOR = (0, 255, 0)
OBJECT_COLOR = (255, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
THICKNESS = 2
LINE_TYPE = cv2.LINE_AA


def draw_results(frame, face_results, object_results):
    canvas = frame.copy()

    for obj in object_results:
        x1, y1, x2, y2 = obj["bbox"]
        label = f"{obj['class_name']} {obj['confidence']:.2f}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), OBJECT_COLOR, THICKNESS)
        _draw_label(canvas, label, (x1, y1), OBJECT_COLOR)

    for face in face_results:
        x1, y1, x2, y2 = face["bbox"]
        gender = face.get("gender") or ""
        name = face["name"]
        label = f"{name} {gender}".strip()
        cv2.rectangle(canvas, (x1, y1), (x2, y2), FACE_COLOR, THICKNESS)
        _draw_label(canvas, label, (x1, y1), FACE_COLOR)

    return canvas


def _draw_label(img, text, pos, color):
    x, y = pos
    if y < 20:
        y = 20
    (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE, THICKNESS)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 4, y + 4), color, -1)
    cv2.putText(img, text, (x + 2, y - 2), FONT, FONT_SCALE, (255, 255, 255), THICKNESS, LINE_TYPE)
