import os
import time

import cv2
import easyocr
from ultralytics import YOLO

from ml.db import save_violation


MODEL_PATH = "runs/detect/train6/weights/best.pt"
PLATE_MODEL_PATH = "yolov8n.pt"
OUTPUT_VIDEO_PATH = "static/output.mp4"
FACES_DIR = "static/outputs/faces"
PLATES_DIR = "static/outputs/plates"
PLATES_TEXT_PATH = "static/outputs/plates/plates.txt"
CONFIDENCE_THRESHOLD = 0.25
DEFAULT_FPS = 20.0
MIN_FRAMES = 10


def _ensure_output_directory():
    os.makedirs(os.path.dirname(OUTPUT_VIDEO_PATH) or ".", exist_ok=True)
    os.makedirs(FACES_DIR, exist_ok=True)
    os.makedirs(PLATES_DIR, exist_ok=True)


def _clear_face_outputs():
    if not os.path.exists(FACES_DIR):
        return
    for filename in os.listdir(FACES_DIR):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            try:
                os.remove(os.path.join(FACES_DIR, filename))
            except OSError:
                pass


def _clear_plate_outputs():
    if not os.path.exists(PLATES_DIR):
        return
    for filename in os.listdir(PLATES_DIR):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".txt")):
            try:
                os.remove(os.path.join(PLATES_DIR, filename))
            except OSError:
                pass


def _normalize_label(raw_label):
    label = str(raw_label).strip().lower().replace("-", "_").replace(" ", "_")
    if label in {"helmet", "with_helmet"}:
        return "helmet"
    if label in {"no_helmet", "nohelmet", "without_helmet"}:
        return "no_helmet"
    return None


def _clip_bbox(x1, y1, x2, y2, width, height):
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(0, min(int(x2), width - 1))
    y2 = max(0, min(int(y2), height - 1))
    return x1, y1, x2, y2


def _draw_detection(frame, bbox, label, confidence):
    color = (0, 200, 0) if label == "helmet" else (0, 0, 255)
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    caption = f"{label} {confidence:.2f}"
    text_y = y1 - 10 if y1 > 24 else y1 + 22
    cv2.putText(
        frame,
        caption,
        (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def process_video(input_video_path):
    start_time = time.time()
    _ensure_output_directory()
    _clear_face_outputs()
    _clear_plate_outputs()

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"YOLO model not found at {MODEL_PATH}")

    model = YOLO(MODEL_PATH)
    plate_model = YOLO(PLATE_MODEL_PATH)
    reader = easyocr.Reader(["en"])
    class_names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    if frame_width <= 0 or frame_height <= 0:
        cap.release()
        raise ValueError("Unable to determine video dimensions.")

    if os.path.exists(OUTPUT_VIDEO_PATH):
        os.remove(OUTPUT_VIDEO_PATH)

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(
        OUTPUT_VIDEO_PATH,
        fourcc,
        fps,
        (frame_width, frame_height),
    )
    if not out.isOpened():
        print("avc1 failed, falling back to mp4v")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            OUTPUT_VIDEO_PATH,
            fourcc,
            fps,
            (frame_width, frame_height),
        )

    if not out.isOpened():
        cap.release()
        raise RuntimeError("Failed to open video writer.")

    track_memory = {}
    saved_face_ids = set()
    saved_faces = []
    saved_plate_ids = set()
    saved_plates = []
    saved_ids = set()
    saved_face_paths = {}
    saved_plate_texts = {}

    save_violation(
        {
            "track_id": 999,
            "violation": True,
            "face_path": "test.jpg",
            "plate_text": "TEST123",
            "timestamp": 0,
        }
    )

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            try:
                results = model.track(
                    frame,
                    persist=True,
                    conf=0.4,
                    iou=0.5,
                    tracker="bytetrack.yaml",
                    verbose=False,
                )
            except Exception:
                results = []

            for r in results:
                boxes = r.boxes
                if boxes is None:
                    continue

                for box in boxes:
                    cls_id = int(box.cls[0])
                    track_id = int(box.id[0]) if box.id is not None else None
                    if track_id is None:
                        continue

                    label = _normalize_label(class_names.get(cls_id, str(cls_id)))
                    if label not in {"helmet", "no_helmet"}:
                        continue

                    conf = float(box.conf[0]) if box.conf is not None else 0.0
                    if conf < 0.6:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1, y1, x2, y2 = _clip_bbox(x1, y1, x2, y2, frame_width, frame_height)
                    if track_id not in track_memory:
                        track_memory[track_id] = {
                            "frames": 1,
                            "labels": [],
                        }
                    else:
                        track_memory[track_id]["frames"] += 1
                    track_memory[track_id]["labels"].append(label)
                    print("Track ID:", track_id)

                    if label == "no_helmet":
                        print("⚠️ VIOLATION DETECTED:", track_id)
                        face_image_path = saved_face_paths.get(track_id)
                        plate_text = saved_plate_texts.get(track_id)

                        if track_id not in saved_face_ids:
                            rider_crop = frame[y1:y2, x1:x2]
                            if rider_crop.size != 0:
                                gray = cv2.cvtColor(rider_crop, cv2.COLOR_BGR2GRAY)
                                gray = cv2.equalizeHist(gray)
                                faces = face_cascade.detectMultiScale(gray, 1.1, 3)
                                print("Faces detected:", len(faces))
                                for (fx, fy, fw, fh) in faces:
                                    face_img = rider_crop[fy:fy + fh, fx:fx + fw]
                                    if face_img.size == 0:
                                        continue
                                    filename = f"face_{track_id}.jpg"
                                    save_path = os.path.join(FACES_DIR, filename)
                                    if cv2.imwrite(save_path, face_img):
                                        face_image_path = f"/static/outputs/faces/{filename}"
                                        saved_faces.append(face_image_path)
                                        saved_face_paths[track_id] = face_image_path
                                        saved_face_ids.add(track_id)
                                    break

                        if track_id not in saved_plate_ids:
                            rider_crop = frame[y1:y2, x1:x2]
                            if rider_crop.size != 0:
                                try:
                                    plate_results = plate_model(rider_crop, verbose=False)
                                except Exception:
                                    plate_results = []

                                for plate_result in plate_results:
                                    if plate_result.boxes is None:
                                        continue

                                    for plate_box in plate_result.boxes:
                                        px1, py1, px2, py2 = map(int, plate_box.xyxy[0])
                                        px1, py1, px2, py2 = _clip_bbox(
                                            px1,
                                            py1,
                                            px2,
                                            py2,
                                            rider_crop.shape[1],
                                            rider_crop.shape[0],
                                        )
                                        plate_crop = rider_crop[py1:py2, px1:px2]
                                        if plate_crop.size == 0:
                                            continue

                                        filename = f"plate_{track_id}.jpg"
                                        save_path = os.path.join(PLATES_DIR, filename)
                                        if cv2.imwrite(save_path, plate_crop):
                                            plate_text = ""
                                            text_results = reader.readtext(plate_crop)
                                            for res in text_results:
                                                plate_text = res[1]
                                                print("Plate:", plate_text)
                                                with open(PLATES_TEXT_PATH, "a", encoding="utf-8") as f:
                                                    f.write(plate_text + "\n")
                                            saved_plate_texts[track_id] = plate_text or None
                                            saved_plates.append(
                                                {
                                                    "image": filename,
                                                    "text": plate_text,
                                                    "track_id": track_id,
                                                }
                                            )
                                            saved_plate_ids.add(track_id)
                                        break

                                    if track_id in saved_plate_ids:
                                        break

                        if track_id not in saved_ids:
                            frame_time = round(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, 2)
                            violation_data = {
                                "track_id": track_id,
                                "violation": True,
                                "face_path": face_image_path if face_image_path else None,
                                "plate_text": plate_text if plate_text else None,
                                "timestamp": frame_time,
                            }
                            print("💾 CALLING SAVE FUNCTION")
                            save_violation(violation_data)
                            saved_ids.add(track_id)

                    _draw_detection(frame, (x1, y1, x2, y2), label, conf)

            out.write(frame)
    finally:
        cap.release()
        out.release()

    total_motorcycles = 0
    total_violations = 0

    for _, data in track_memory.items():
        if data["frames"] < MIN_FRAMES:
            continue

        total_motorcycles += 1
        if data["labels"].count("no_helmet") > data["labels"].count("helmet"):
            total_violations += 1

    processing_time = round(time.time() - start_time, 2)
    print("Saved video at:", OUTPUT_VIDEO_PATH)
    print("Video exists:", os.path.exists(OUTPUT_VIDEO_PATH))
    print("Video saved:", os.path.exists(OUTPUT_VIDEO_PATH))
    print("Video size:", os.path.getsize(OUTPUT_VIDEO_PATH) if os.path.exists(OUTPUT_VIDEO_PATH) else 0)
    print("Total riders:", total_motorcycles)
    print("Total violations:", total_violations)
    print("Saved faces:", len(saved_face_ids))

    stats = {
        "motorcycles": total_motorcycles,
        "violations": total_violations,
        "total_motorcycles": total_motorcycles,
        "helmet_violations": total_violations,
        "plates": 0,
    }
    faces = saved_faces
    plates = saved_plates
    timeline = []

    return {
        "video": "/static/output.mp4",
        "processed_video": "static/output.mp4",
        "output_video": "static/output.mp4",
        "stats": stats,
        "faces": faces,
        "plates": plates,
        "timeline": timeline,
        "processing_time": processing_time,
    }
