"""
Download required model weights for RoboCup vision pipeline.

Downloads:
  1. InsightFace buffalo_l face model from ModelScope
  2. YOLO yolo11n.pt for object detection (into repo models/)

Usage:
    python scripts/download_models.py
"""

import hashlib
import os
import platform
import shutil
import sys
import zipfile
from pathlib import Path


BUFFALO_L_SHA256 = "80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f"
YOLO_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
YOLO_MODEL_ID = "yolo11n.pt"


def _info(msg):
    print(f"[INFO] {msg}")


def _error(msg):
    print(f"[ERROR] {msg}", file=sys.stderr)


def _ok(msg):
    print(f"[OK] {msg}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
#  1. InsightFace buffalo_l
# ---------------------------------------------------------------------------

def _ensure_modelscope():
    try:
        import modelscope  # noqa
    except ImportError:
        _info("installing modelscope ...")
        import subprocess
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-U", "modelscope"]
        )


def _download_buffalo_l(insightface_root: Path) -> Path:
    zip_path = insightface_root / "buffalo_l.zip"
    if zip_path.exists() and _sha256(zip_path) == BUFFALO_L_SHA256:
        _ok(f"buffalo_l.zip already downloaded and verified ({zip_path})")
        return zip_path
    _info("downloading buffalo_l from ModelScope (deepghs/insightface) ...")
    _ensure_modelscope()
    from modelscope.hub.snapshot_download import snapshot_download
    ms_path = snapshot_download(
        model_id="deepghs/insightface",
        local_dir=str(insightface_root / "modelscope_tmp"),
        cache_dir=str(insightface_root / "modelscope_cache"),
    )
    _info(f"ModelScope download path: {ms_path}")
    candidates = list(Path(ms_path).rglob("*buffalo_l*"))
    src = None
    for c in candidates:
        if c.suffix == ".zip":
            src = c
            break
    if src is None:
        raise FileNotFoundError(
            f"buffalo_l.zip not found in ModelScope download ({ms_path})"
        )
    _info(f"found buffalo_l.zip at {src}")
    if zip_path.exists():
        zip_path.unlink()
    shutil.move(str(src), str(zip_path))
    shutil.rmtree(insightface_root / "modelscope_tmp", ignore_errors=True)
    shutil.rmtree(insightface_root / "modelscope_cache", ignore_errors=True)
    actual = _sha256(zip_path)
    if actual != BUFFALO_L_SHA256:
        raise RuntimeError(
            f"SHA-256 mismatch for buffalo_l.zip\n"
            f"  expected: {BUFFALO_L_SHA256}\n"
            f"  actual:   {actual}"
        )
    _ok("buffalo_l.zip verified")
    return zip_path


def _extract_buffalo_l(zip_path: Path, target_dir: Path):
    if target_dir.is_dir() and any(target_dir.iterdir()):
        _ok(f"buffalo_l already extracted at {target_dir}")
        return
    _info(f"extracting buffalo_l.zip to {target_dir} ...")
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(str(target_dir))
    nested = target_dir / "buffalo_l"
    if nested.is_dir():
        for item in nested.iterdir():
            shutil.move(str(item), str(target_dir / item.name))
        shutil.rmtree(nested)
    expected = ["1k3d68.onnx", "2d106det.onnx", "det_10g.onnx",
                "genderage.onnx", "w600k_r50.onnx"]
    found = [f.name for f in target_dir.iterdir() if f.suffix == ".onnx"]
    for name in expected:
        if name not in found:
            raise FileNotFoundError(
                f"missing ONNX file after extraction: {name}"
            )
    _ok(f"buffalo_l extracted ({len(found)} ONNX files)")


def download_insightface():
    root = Path.home() / ".insightface"
    model_dir = root / "models" / "buffalo_l"
    if model_dir.is_dir() and any(model_dir.iterdir()):
        _ok(f"InsightFace buffalo_l already installed at {model_dir}")
        return
    zip_path = _download_buffalo_l(root)
    _extract_buffalo_l(zip_path, model_dir)
    zip_path.unlink(missing_ok=True)
    _ok("InsightFace buffalo_l ready")


# ---------------------------------------------------------------------------
#  2. YOLO yolo11n.pt
# ---------------------------------------------------------------------------

def download_yolo():
    repo_root = Path(__file__).resolve().parent.parent
    dest = repo_root / "models" / YOLO_MODEL_ID
    if dest.exists():
        _ok(f"YOLO model already exists at {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    _info(f"downloading {YOLO_MODEL_ID} ...")
    import urllib.request
    import tempfile
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
    try:
        urllib.request.urlretrieve(YOLO_URL, tmp.name)
        shutil.move(tmp.name, str(dest))
    except Exception:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
        raise
    size_mb = dest.stat().st_size / (1024 * 1024)
    _ok(f"{YOLO_MODEL_ID} downloaded ({size_mb:.1f} MB) to {dest}")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    _info(f"platform: {platform.system()} {platform.machine()}")
    _info(f"python:   {sys.version}")
    print()
    download_insightface()
    print()
    download_yolo()
    print()
    _ok("all models downloaded successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
