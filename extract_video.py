import os

import cv2

video_path = "C:/Users/USER/Desktop/Competition_rmuti_17/video.mp4"
output_dir = "C:/Users/USER/Desktop/Competition_rmuti_17/extract_video"
frame_interval = 30 -- # Extract every 30th frame

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

count = 0
saved = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if count % frame_interval == 0:
        cv2.imwrite(f"{output_dir}/frame_{saved:04d}.jpg", frame)
        saved += 1

    count += 1

cap.release()
