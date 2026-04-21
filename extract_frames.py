import cv2
import os

video_folder = "training_data/videos"
output_folder = "training_data/images"

os.makedirs(output_folder, exist_ok=True)

frame_skip = 10  
video_count = 0
image_count = 0

for video in os.listdir(video_folder):

    video_path = os.path.join(video_folder, video)
    cap = cv2.VideoCapture(video_path)

    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % frame_skip == 0:
            image_name = f"frame_{video_count}_{frame_number}.jpg"
            cv2.imwrite(os.path.join(output_folder, image_name), frame)
            image_count += 1

        frame_number += 1

    cap.release()
    video_count += 1

print("Extraction finished")
print("Total images created:", image_count)