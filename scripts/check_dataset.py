"""
YOLO dataset validation script.

Usage:
    python -m scripts.check_dataset data/processed/item_dataset/data.yaml
"""

import hashlib
import sys
from collections import Counter
from pathlib import Path

import yaml


def _file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _list_images(dir_path: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted([p for p in sorted(dir_path.rglob("*")) if p.suffix.lower() in exts])


def _list_labels(dir_path: Path) -> list[Path]:
    return sorted([p for p in dir_path.rglob("*.txt")])


def _label_path(image_path: Path, split_dir: Path, label_dir: Path) -> Path:
    rel = image_path.relative_to(split_dir)
    return label_dir / rel.with_suffix(".txt")


def _image_path(label_path: Path, split_dir: Path, label_dir: Path) -> Path:
    rel = label_path.relative_to(label_dir)
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        candidate = split_dir / rel.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def check_dataset(yaml_path_str: str):
    yaml_path = Path(yaml_path_str)
    if not yaml_path.exists():
        print(f"ERROR: {yaml_path} not found")
        sys.exit(1)

    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    required_keys = ["train", "val", "names"]
    for k in required_keys:
        if k not in cfg:
            print(f"ERROR: data.yaml missing key '{k}'")
            sys.exit(1)

    names = cfg["names"]
    nc = cfg.get("nc", len(names))
    if len(names) != nc:
        print(f"ERROR: names count ({len(names)}) != nc ({nc})")
        sys.exit(1)

    print(f"Classes ({nc}): {names}")
    print()

    base = yaml_path.parent
    errors = 0
    background_count = 0
    class_counts: Counter = Counter()
    total_images = 0
    total_labels = 0
    train_hashes = {}
    val_hashes = {}
    all_image_names = {}

    for split in ["train", "val", "test"]:
        split_path = cfg.get(split)
        if not split_path:
            if split == "test":
                print(f"  [{split}] not configured (optional, skipping)")
                continue
            print(f"ERROR: [{split}] not configured")
            errors += 1
            continue

        split_dir = base / split_path
        if not split_dir.is_dir():
            print(f"ERROR: [{split}] directory not found: {split_dir}")
            errors += 1
            continue

        label_dir = base / "labels" / split
        images = _list_images(split_dir)
        total_images += len(images)
        all_image_names[split] = {p.name for p in images}
        print(f"[{split}] {len(images)} images")

        for img_path in images:
            lbl = _label_path(img_path, split_dir, label_dir)
            if not lbl.exists():
                print(f"  WARN: missing label: {lbl}")
                errors += 1
                continue

            lines = lbl.read_text().strip().splitlines()
            if not lines:
                background_count += 1
                continue

            total_labels += len(lines)
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    print(f"  ERROR: {lbl}: expected 5 values, got {len(parts)}: {line}")
                    errors += 1
                    continue
                try:
                    cls_id = int(parts[0])
                except ValueError:
                    print(f"  ERROR: {lbl}: invalid class_id: {parts[0]}")
                    errors += 1
                    continue
                if cls_id < 0 or cls_id >= nc:
                    print(f"  ERROR: {lbl}: class_id {cls_id} out of range [0, {nc - 1}]")
                    errors += 1
                    continue
                class_counts[cls_id] += 1

                vals = []
                valid = True
                for i, val_str in enumerate(parts[1:], 1):
                    try:
                        v = float(val_str)
                    except ValueError:
                        print(f"  ERROR: {lbl}: invalid value at position {i}: {val_str}")
                        errors += 1
                        valid = False
                        break
                    if v < 0 or v > 1:
                        print(f"  ERROR: {lbl}: value at position {i} ({v}) outside [0, 1]")
                        errors += 1
                        valid = False
                        break
                    vals.append(v)
                if not valid:
                    continue
                _, _, w, h = vals
                if w <= 0 or h <= 0:
                    print(f"  ERROR: {lbl}: width ({w}) and height ({h}) must be > 0")
                    errors += 1

            h = _file_hash(img_path)
            if split == "train":
                train_hashes[h] = img_path
            elif split == "val":
                val_hashes[h] = img_path

        orphaned_labels = []
        for lbl_path in _list_labels(label_dir):
            if not _image_path(lbl_path, split_dir, label_dir):
                orphaned_labels.append(lbl_path)
        if orphaned_labels:
            for ol in orphaned_labels:
                print(f"  WARN: orphaned label (no matching image): {ol}")
            errors += len(orphaned_labels)

    if train_hashes and val_hashes:
        common = set(train_hashes) & set(val_hashes)
        if common:
            for h in sorted(common):
                print(f"  WARN: identical file in train and val: {train_hashes[h].name}")
            errors += len(common)

    print()
    if background_count:
        print(f"Background images (empty labels): {background_count}")
    if errors:
        print(f"Found {errors} issue(s)")
    else:
        print("No issues found")

    if total_labels:
        print(f"\nClass distribution ({total_labels} labels across {total_images} images):")
        max_count = max(class_counts.values()) if class_counts else 1
        for cls_id in range(nc):
            count = class_counts.get(cls_id, 0)
            bar_len = int(count / max_count * 40)
            bar = "█" * bar_len
            print(f"  {names[cls_id]:20s} ({cls_id}): {count:5d} {bar}")

    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <data.yaml>")
        sys.exit(1)
    check_dataset(sys.argv[1])
