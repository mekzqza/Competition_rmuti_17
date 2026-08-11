import sys
from pathlib import Path

import cv2

video_path = sys.argv[1] if len(sys.argv) > 1 else "video.mp4"
out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "extract_video")
every = int(sys.argv[3]) if len(sys.argv) > 3 else 30

out_dir.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), f"cannot open video: {video_path}"

read = saved = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if read % every == 0:
        cv2.imwrite(str(out_dir / f"frame_{saved:04d}.jpg"), frame)
        saved += 1
    read += 1

cap.release()
print(f"read {read} frames -> saved {saved} imgs to {out_dir.resolve()}")
