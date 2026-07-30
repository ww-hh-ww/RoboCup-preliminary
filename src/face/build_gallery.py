import json
import re
import numpy as np
from pathlib import Path


GALLERY_DIR = Path("data/processed/face_gallery")
EMBEDDING_DIM = 512

GALLERY_SPEC = [
    {"person_id": "person_001", "name": "张三", "gender": "男"},
    {"person_id": "person_002", "name": "李四", "gender": "男"},
    {"person_id": "person_003", "name": "王婷", "gender": "女"},
    {"person_id": "person_004", "name": "赵雷", "gender": "男"},
    {"person_id": "person_005", "name": "陈雨", "gender": "女"},
]

_EMBEDDING_PATTERN = re.compile(r"^embedding.*\.npy$")


def _normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


class FaceGallery:

    def __init__(self, gallery_dir=None):
        self.gallery_dir = Path(gallery_dir) if gallery_dir else GALLERY_DIR
        self.embeddings: dict[str, list[np.ndarray]] = {}
        self.metadata: dict[str, dict] = {}
        self._load()

    def _load(self):
        if not self.gallery_dir.is_dir():
            return
        for person_dir in sorted(self.gallery_dir.iterdir()):
            if not person_dir.is_dir():
                continue
            pid = person_dir.name
            meta_path = person_dir / "metadata.json"
            if meta_path.exists():
                self.metadata[pid] = json.loads(meta_path.read_text(encoding="utf-8"))
            emb_list = []
            for emb_path in sorted(person_dir.iterdir()):
                if _EMBEDDING_PATTERN.match(emb_path.name):
                    emb_list.append(np.load(str(emb_path)))
            if emb_list:
                self.embeddings[pid] = emb_list

    def match(self, image_or_embedding, threshold=0.4):
        if isinstance(image_or_embedding, (str, Path)):
            emb = self._extract(image_or_embedding)
            if emb is None:
                return {"person_id": None, "name": "Unknown", "gender": None,
                        "confidence": 0.0, "top2_similarity": 0.0, "match_status": "unknown"}
        else:
            emb = _normalize(np.asarray(image_or_embedding, dtype=np.float32))

        best_id = None
        best_sim = -1.0
        second_id = None
        second_sim = -1.0
        for pid, embeddings in self.embeddings.items():
            for gallery_emb in embeddings:
                sim = float(np.dot(emb, gallery_emb))
                if sim > best_sim:
                    second_id = best_id
                    second_sim = best_sim
                    best_sim = sim
                    best_id = pid
                elif sim > second_sim:
                    second_id = pid
                    second_sim = sim

        if best_id is None or best_sim < threshold:
            return {"person_id": None, "name": "Unknown", "gender": None,
                    "confidence": best_sim, "top2_person_id": None,
                    "top2_name": None, "top2_similarity": second_sim,
                    "match_status": "unknown"}

        meta = self.metadata.get(best_id, {})
        second_meta = self.metadata.get(second_id) if second_id else None
        return {
            "person_id": best_id,
            "name": meta.get("name", "Unknown"),
            "gender": meta.get("gender"),
            "confidence": best_sim,
            "top2_person_id": second_id,
            "top2_name": second_meta.get("name") if second_meta else None,
            "top2_similarity": second_sim,
            "match_status": "known",
        }

    def add_person(self, person_id, name, gender, embedding, filename=None):
        person_dir = self.gallery_dir / person_id
        person_dir.mkdir(parents=True, exist_ok=True)
        metadata = {"person_id": person_id, "name": name, "gender": gender}
        (person_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if filename:
            emb_name = f"embedding_{filename}.npy"
        else:
            existing = [p for p in person_dir.iterdir() if _EMBEDDING_PATTERN.match(p.name)]
            idx = len(existing) + 1
            emb_name = f"embedding_{idx:03d}.npy"
        np.save(str(person_dir / emb_name), _normalize(np.asarray(embedding, dtype=np.float32)))
        self.metadata[person_id] = metadata
        if person_id not in self.embeddings:
            self.embeddings[person_id] = []
        self.embeddings[person_id].append(_normalize(np.asarray(embedding, dtype=np.float32)))

    def _extract(self, image_path):
        faces = self._extract_faces(image_path)
        if faces is None:
            return None
        return _normalize(faces[0].normed_embedding)

    def _extract_faces(self, image_path):
        try:
            import cv2
            import insightface
            from insightface.app import FaceAnalysis
        except ImportError:
            raise ImportError("insightface and opencv-python required for real encoding")
        app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
            root=Path.home() / ".insightface",
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        faces = app.get(img)
        if not faces:
            return None
        return faces


def build_simulated_gallery(output_dir=None, seed=42):
    rng = np.random.default_rng(seed)
    gallery_dir = Path(output_dir) if output_dir else GALLERY_DIR
    if gallery_dir.is_dir():
        import shutil
        shutil.rmtree(gallery_dir)
    gallery = FaceGallery(str(gallery_dir))
    for spec in GALLERY_SPEC:
        emb = rng.normal(size=EMBEDDING_DIM).astype(np.float32)
        gallery.add_person(spec["person_id"], spec["name"], spec["gender"], emb)
    print(f"Gallery built at {gallery_dir} ({len(GALLERY_SPEC)} known)")
    return gallery


def build_real_gallery(input_dir, output_dir=None):
    input_dir = Path(input_dir)
    gallery_dir = Path(output_dir) if output_dir else GALLERY_DIR
    gallery = FaceGallery(str(gallery_dir))
    for person_dir in sorted(input_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        meta_path = person_dir / "metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        pid = meta["person_id"]
        images_dir = person_dir / "images"
        if not images_dir.is_dir():
            images_dir = person_dir
        added = False
        for img_path in sorted(images_dir.glob("*")):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            faces = gallery._extract_faces(img_path)
            if faces is None:
                print(f"  ! {pid}: {img_path.name}: no face detected, skipping")
                continue
            if len(faces) > 1:
                raise ValueError(
                    f"{pid} ({img_path.name}): {len(faces)} faces detected, "
                    f"expected exactly 1"
                )
            emb = _normalize(faces[0].normed_embedding)
            gallery.add_person(pid, meta.get("name", ""), meta.get("gender"), emb,
                               filename=img_path.stem)
            print(f"  + {pid}: {img_path.name}")
            added = True
        if not added:
            print(f"  ! {pid}: no valid face photo found, skipping")
    print(f"Gallery built at {gallery_dir}")
    return gallery


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "simulate"
    if mode == "simulate":
        build_simulated_gallery()
    elif mode == "real":
        input_dir = sys.argv[2] if len(sys.argv) > 2 else "data/raw/faces"
        build_real_gallery(input_dir)
    else:
        print(f"Usage: {sys.argv[0]} [simulate|real <input_dir>]")
