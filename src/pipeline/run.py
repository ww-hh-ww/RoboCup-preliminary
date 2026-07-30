import argparse
import cv2
from pathlib import Path

from src.frame_source import ImageFrameSource, CameraFrameSource
from src.face.detector import FaceDetector
from src.object.detector import ObjectDetector
from src.visualize.draw import draw_results


def main():
    parser = argparse.ArgumentParser(description="Robocup vision pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to input image")
    group.add_argument("--camera", type=int, nargs="?", const=0, help="Camera device ID")
    parser.add_argument("--output", type=str, help="Output image path (image mode only)")
    parser.add_argument("--face-threshold", type=float, default=0.4, help="Face match threshold")
    parser.add_argument("--obj-threshold", type=float, default=0.25, help="Object detection confidence")
    parser.add_argument("--no-display", action="store_true", help="Don't show window")
    args = parser.parse_args()

    if args.image:
        source = ImageFrameSource(args.image)
    else:
        source = CameraFrameSource(args.camera or 0)

    face_detector = FaceDetector(threshold=args.face_threshold)
    obj_detector = ObjectDetector()

    for i, frame in enumerate(source):
        faces = face_detector.detect(frame)
        objects = obj_detector.detect(frame, conf_threshold=args.obj_threshold)

        print(f"Frame {i}: {len(faces)} face(s), {len(objects)} object(s)")
        for f in faces:
            print(f"  Face: {f['name']} ({f['gender']}) conf={f['confidence']:.2f}")
        for o in objects:
            print(f"  Object: {o['class_name']} conf={o['confidence']:.2f}")

        output = draw_results(frame, faces, objects)

        if args.output and args.image:
            cv2.imwrite(args.output, output)
            print(f"Saved to {args.output}")

        if not args.no_display:
            cv2.imshow("Robocup Vision", output)
            key = cv2.waitKey(1 if args.camera is not None else 0)
            if key == 27 or key == ord("q"):
                break

    source.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
