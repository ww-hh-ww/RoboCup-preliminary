from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class FaceResult:
    bbox: list[int]
    name: str
    gender: Optional[str]
    confidence: float
    face_confidence: float
    top1_name: Optional[str]
    top1_similarity: float
    top2_name: Optional[str]
    top2_similarity: float
    match_status: str


@dataclass
class ObjectResult:
    bbox: list[int]
    class_id: int
    class_name: str
    confidence: float


@dataclass
class DetectionResult:
    image_path: str
    faces: list[FaceResult]
    objects: list[ObjectResult]
    status: str
    error_message: str = ""


def result_to_dict(r: DetectionResult) -> dict:
    d = asdict(r)
    d["image"] = d.pop("image_path")
    d.pop("status", None)
    d.pop("error_message", None)
    return d
