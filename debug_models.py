import os
import random
from pathlib import Path

import cv2
from ultralytics import YOLO


MODEL_PATHS = [
    "runs/detect/train3/weights/best.pt",
    "runs/detect/train4/weights/best.pt",
]
IMAGES_DIR = Path("dataset/images")
FIXED_IMAGE = IMAGES_DIR / "frame_0_10.jpg"
PREFERRED_VIDEO = Path("your_video.mp4")
VIDEO_FALLBACK_DIR = Path("static/uploads")
MAX_RANDOM_IMAGES = 3
MAX_VIDEO_FRAMES = 30
CONFIDENCE = 0.1


def format_size(size_bytes):
    units = ["B", "KB", "MB", "GB"]
    size = float(size_bytes)
    unit_index = 0
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1
    return f"{size:.2f} {units[unit_index]}"


def choose_images():
    if not IMAGES_DIR.exists():
        return []

    image_paths = sorted(
        path for path in IMAGES_DIR.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_paths:
        return []

    random_images = random.sample(image_paths, min(MAX_RANDOM_IMAGES, len(image_paths)))
    selected = []

    if FIXED_IMAGE.exists():
        selected.append(FIXED_IMAGE)

    for image_path in random_images:
        if image_path not in selected:
            selected.append(image_path)

    return selected


def choose_video():
    if PREFERRED_VIDEO.exists():
        return PREFERRED_VIDEO

    if VIDEO_FALLBACK_DIR.exists():
        candidates = sorted(
            path
            for path in VIDEO_FALLBACK_DIR.iterdir()
            if path.is_file() and path.suffix.lower() in {".mp4", ".mov"}
        )
        if candidates:
            return candidates[0]

    return PREFERRED_VIDEO


def print_image_results(model, image_path):
    if not image_path.exists():
        print(f"  {image_path.name} -> missing, skipped")
        return

    try:
        results = model(str(image_path), conf=CONFIDENCE, verbose=False)
    except Exception as exc:
        print(f"  {image_path.name} -> inference error: {exc}")
        return

    result = results[0]
    boxes = result.boxes
    detection_count = len(boxes) if boxes is not None else 0
    print(f"  {image_path.name} -> {detection_count} detections")

    if boxes is None:
        return

    classes = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else []
    confidences = boxes.conf.cpu().numpy() if boxes.conf is not None else []

    for class_id, confidence in zip(classes, confidences):
        class_name = model.names.get(int(class_id), str(class_id))
        print(f"    - {class_name} ({float(confidence):.2f})")


def print_video_results(model, video_path):
    print("Video test:")

    if not video_path.exists():
        print(f"  Video missing: {video_path}")
        return

    print(f"  Using video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("  Unable to open video")
        return

    frame_index = 0

    try:
        while frame_index < MAX_VIDEO_FRAMES:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("  Frame read failed or end of video reached")
                break

            frame_index += 1

            try:
                results = model(frame, conf=CONFIDENCE, verbose=False)
            except Exception as exc:
                print(f"  Frame {frame_index} -> inference error: {exc}")
                continue

            result = results[0]
            boxes = result.boxes
            detection_count = len(boxes) if boxes is not None else 0
            print(f"  Frame {frame_index} -> {detection_count} detections")

            if boxes is None:
                continue

            classes = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else []
            confidences = boxes.conf.cpu().numpy() if boxes.conf is not None else []
            for class_id, confidence in zip(classes, confidences):
                class_name = model.names.get(int(class_id), str(class_id))
                print(f"    - {class_name} ({float(confidence):.2f})")
    finally:
        cap.release()


def evaluate_model(model_path, image_paths, video_path):
    model_name = Path(model_path).parts[-3] if len(Path(model_path).parts) >= 3 else Path(model_path).stem
    print(f"======== MODEL: {model_name} ========")
    print(f"Model path: {model_path}")

    if not os.path.exists(model_path):
        print(f"ERROR: model file missing -> {model_path}")
        print()
        return

    file_size = os.path.getsize(model_path)
    print(f"File size: {format_size(file_size)}")

    try:
        model = YOLO(model_path)
    except Exception as exc:
        print(f"ERROR: failed to load model -> {exc}")
        print()
        return

    print(f"Class names: {model.names}")
    print("Image test:")
    if not image_paths:
        print("  No images available")
    else:
        for image_path in image_paths:
            print_image_results(model, image_path)

    print_video_results(model, video_path)
    print()


def main():
    random.seed(42)
    image_paths = choose_images()
    video_path = choose_video()

    for model_path in MODEL_PATHS:
        evaluate_model(model_path, image_paths, video_path)


if __name__ == "__main__":
    main()
