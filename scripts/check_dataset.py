"""
YOLO dataset validation script.

Usage:
    python -m scripts.check_dataset data/processed/item_dataset/data.yaml

Checks:
    - data.yaml exists and is valid
    - train/val/test directories exist
    - Each image has a matching label file
    - Label format (class_id x_center y_center width height, normalized)
    - Class IDs are within [0, nc)
    - Class balance report
"""

import sys
from collections import Counter
from pathlib import Path

import yaml


def _list_images(dir_path: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted([p for p in dir_path.rglob("*") if p.suffix.lower() in exts])


def _label_path(image_path: Path, label_dir: Path) -> Path:
    return label_dir / image_path.relative_to(image_path.parent).with_suffix(".txt")


def check_dataset(yaml_path_str: str):
    yaml_path = Path(yaml_path_str)
    if not yaml_path.exists():
        print(f"ERROR: {yaml_path} not found")
        sys.exit(1)

    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    required_keys = ["train", "val", "nc", "names"]
    for k in required_keys:
        if k not in cfg:
            print(f"ERROR: data.yaml missing key '{k}'")
            sys.exit(1)

    nc = cfg["nc"]
    names = cfg["names"]
    if len(names) != nc:
        print(f"ERROR: names count ({len(names)}) != nc ({nc})")
        sys.exit(1)

    print(f"Classes ({nc}): {names}")
    print()

    base = yaml_path.parent
    errors = 0
    class_counts: Counter = Counter()
    total_images = 0
    total_labels = 0

    for split in ["train", "val", "test"]:
        split_dir = base / cfg[split]
        if not split_dir.exists():
            if split == "test":
                print(f"  [{split}] directory not found (optional, skipping)")
                continue
            print(f"ERROR: [{split}] directory not found: {split_dir}")
            errors += 1
            continue

        label_dir = base / "labels" / split
        images = _list_images(split_dir)
        total_images += len(images)
        print(f"[{split}] {len(images)} images")

        for img_path in images:
            lbl = _label_path(img_path, label_dir)
            if not lbl.exists():
                print(f"  WARN: missing label: {lbl}")
                errors += 1
                continue

            lines = lbl.read_text().strip().splitlines()
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

                for i, val in enumerate(parts[1:], 1):
                    try:
                        v = float(val)
                    except ValueError:
                        print(f"  ERROR: {lbl}: invalid value at position {i}: {val}")
                        errors += 1
                        break
                    if v < 0 or v > 1:
                        print(f"  WARN: {lbl}: value at position {i} ({v}) outside [0, 1]")

    print()
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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <data.yaml>")
        sys.exit(1)
    check_dataset(sys.argv[1])
