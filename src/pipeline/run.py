import argparse
import csv
import json
import sys
from pathlib import Path

import cv2

from src.frame_source import ImageFrameSource, CameraFrameSource
from src.face.detector import FaceDetector
from src.object.detector import ObjectDetector
from src.pipeline.types import FaceResult, ObjectResult, DetectionResult, result_to_dict
from src.visualize.draw import draw_results

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def collect_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    result = []
    for p in sorted(path.rglob("*")):
        if p.suffix.lower() in _IMAGE_EXTENSIONS:
            result.append(p)
    return result


def process_single_image(
    frame, image_path: str,
    face_detector, obj_detector, obj_threshold: float,
) -> DetectionResult:
    try:
        faces = face_detector.detect(frame) if face_detector else []
        objects = obj_detector.detect(frame, conf_threshold=obj_threshold) if obj_detector else []

        face_results = [
            FaceResult(
                bbox=f["bbox"],
                name=f["name"],
                gender=f.get("gender"),
                confidence=f["confidence"],
                face_confidence=f["face_confidence"],
                top1_name=f.get("top1_name"),
                top1_similarity=f["top1_similarity"],
                top2_name=f.get("top2_name"),
                top2_similarity=f["top2_similarity"],
                match_status=f["match_status"],
            )
            for f in faces
        ]
        object_results = [
            ObjectResult(
                bbox=o["bbox"],
                class_id=o["class_id"],
                class_name=o["class_name"],
                confidence=o["confidence"],
            )
            for o in objects
        ]
        return DetectionResult(
            image_path=image_path,
            faces=face_results,
            objects=object_results,
            status="success",
        )
    except Exception as e:
        return DetectionResult(
            image_path=image_path,
            faces=[],
            objects=[],
            status="error",
            error_message=str(e),
        )


def save_annotated(frame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), frame)


def save_json(result: DetectionResult, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result_to_dict(result), ensure_ascii=False, indent=2), encoding="utf-8")


def _relative_stem(img_path: Path, input_root: Path) -> str:
    try:
        rel = img_path.relative_to(input_root)
    except ValueError:
        rel = img_path.name
    return str(rel.parent / rel.stem) if rel.parent != Path(".") else rel.stem


def run_batch(image_paths, output_dir, face_detector, obj_detector, obj_threshold, input_root):
    output_dir = Path(output_dir)
    images_out = output_dir / "images"
    json_out = output_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for img_path in image_paths:
        print(f"Processing: {img_path}")
        try:
            source = ImageFrameSource(str(img_path))
            frame = source.read()
            if frame is None:
                raise RuntimeError("Failed to read image")
            result = process_single_image(frame, str(img_path), face_detector, obj_detector, obj_threshold)
        except Exception as e:
            results.append(DetectionResult(
                image_path=str(img_path), faces=[], objects=[],
                status="error", error_message=str(e),
            ))
            continue

        results.append(result)

        if result.status == "success":
            annotated = draw_results(frame, [f.__dict__ for f in result.faces],
                                     [o.__dict__ for o in result.objects])
            stem = _relative_stem(img_path, input_root)
            save_annotated(annotated, images_out / f"{stem}.jpg")
            save_json(result, json_out / f"{stem}.json")

    _write_summary(results, output_dir)
    return results


def run_camera(source, face_detector, obj_detector, obj_threshold, no_display):
    for i, frame in enumerate(source):
        result = process_single_image(frame, f"camera_frame_{i:04d}", face_detector, obj_detector, obj_threshold)

        print(f"Frame {i}: {len(result.faces)} face(s), {len(result.objects)} object(s)")
        for f in result.faces:
            print(f"  Face: {f.name} ({f.gender}) sim={f.confidence:.2f} [{f.match_status}]")
        for o in result.objects:
            print(f"  Object: {o.class_name} conf={o.confidence:.2f}")

        output = draw_results(frame, [f.__dict__ for f in result.faces],
                              [o.__dict__ for o in result.objects])

        if not no_display:
            cv2.imshow("Robocup Vision", output)
            key = cv2.waitKey(1)
            if key == 27 or key == ord("q"):
                break


def _write_summary(results, output_dir: Path):
    summary_path = output_dir / "summary.csv"
    failures_path = output_dir / "failures.csv"

    successes = [r for r in results if r.status == "success"]
    failures = [r for r in results if r.status == "error"]

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "status", "num_faces", "num_objects", "error_message"])
        for r in results:
            writer.writerow([r.image_path, r.status, len(r.faces), len(r.objects), r.error_message])

    with open(failures_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "error_message"])
        for r in failures:
            writer.writerow([r.image_path, r.error_message])

    print(f"\n{'='*40}")
    print(f"Batch complete: {len(successes)} succeeded, {len(failures)} failed")
    print(f"Summary:  {summary_path}")
    if failures:
        print(f"Failures: {failures_path}")
    print(f"{'='*40}")


def main():
    parser = argparse.ArgumentParser(description="Robocup vision pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to input image or directory")
    group.add_argument("--camera", type=int, nargs="?", const=0, help="Camera device ID")
    parser.add_argument("--output", type=str, default=None, help="Output directory (batch) or file path (single image)")
    parser.add_argument("--face-threshold", type=float, default=0.4, help="Face match threshold")
    parser.add_argument("--obj-threshold", type=float, default=0.25, help="Object detection confidence")
    parser.add_argument("--no-display", action="store_true", help="Don't show window")
    parser.add_argument("--no-face", action="store_true", help="Disable face detection")
    parser.add_argument("--no-object", action="store_true", help="Disable object detection")
    args = parser.parse_args()

    face_detector = None if args.no_face else FaceDetector(threshold=args.face_threshold)
    obj_detector = None if args.no_object else ObjectDetector()

    if args.image:
        image_paths = collect_images(Path(args.image))
        if not image_paths:
            print(f"No images found: {args.image}", file=sys.stderr)
            sys.exit(1)

        output_dir = args.output or "output"
        run_batch(image_paths, output_dir, face_detector, obj_detector, args.obj_threshold, Path(args.image))
    else:
        source = CameraFrameSource(args.camera or 0)
        try:
            run_camera(source, face_detector, obj_detector, args.obj_threshold, args.no_display)
        finally:
            source.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
