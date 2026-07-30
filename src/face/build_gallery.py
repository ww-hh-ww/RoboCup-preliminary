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
_FACE_APP = None


def _get_face_app():
    global _FACE_APP
    if _FACE_APP is None:
        import cv2
        from insightface.app import FaceAnalysis
        _FACE_APP = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
            root=Path.home() / ".insightface",
        )
        _FACE_APP.prepare(ctx_id=0, det_size=(640, 640))
    return _FACE_APP


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
                        "confidence": 0.0, "top1_name": None,
                        "top2_name": None, "top2_similarity": 0.0,
                        "match_status": "unknown"}
        else:
            emb = _normalize(np.asarray(image_or_embedding, dtype=np.float32))

        person_sims = []
        for pid, embeddings in self.embeddings.items():
            best = max(float(np.dot(emb, gallery_emb)) for gallery_emb in embeddings)
            person_sims.append((best, pid))
        person_sims.sort(key=lambda x: x[0], reverse=True)

        best_sim = person_sims[0][0] if person_sims else -1.0
        best_id = person_sims[0][1] if person_sims else None
        second_sim = person_sims[1][0] if len(person_sims) > 1 else -1.0
        second_id = person_sims[1][1] if len(person_sims) > 1 else None

        meta = self.metadata.get(best_id) if best_id else None
        second_meta = self.metadata.get(second_id) if second_id else None

        return {
            "person_id": best_id,
            "name": meta["name"] if meta else "Unknown",
            "gender": meta["gender"] if meta else None,
            "confidence": best_sim,
            "top1_name": meta["name"] if meta else None,
            "top2_name": second_meta["name"] if second_meta else None,
            "top2_similarity": second_sim,
            "match_status": "known" if best_sim >= threshold else "unknown",
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
        import cv2
        app = _get_face_app()
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
    if gallery_dir.is_dir():
        import shutil
        shutil.rmtree(gallery_dir)
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
