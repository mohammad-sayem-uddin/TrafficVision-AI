from ultralytics import YOLO
import cv2
import easyocr

# Load pretrained model (temporary use general model)
model = YOLO("yolov8n.pt")
reader = easyocr.Reader(['en'])

img = cv2.imread("test.jpg")

results = model(img)

for r in results:
    boxes = r.boxes
    if boxes is None:
        continue

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

ocr_results = reader.readtext(img)

for res in ocr_results:
    print("Detected text:", res[1])

cv2.imshow("Test", img)
cv2.waitKey(0)
