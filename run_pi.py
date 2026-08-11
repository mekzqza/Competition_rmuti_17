"""รันบน Raspberry Pi + วัด FPS (ขั้น 8)

    python run_pi.py best.onnx test.mp4          # ไฟล์วิดีโอ
    python run_pi.py best.onnx 0 --show          # กล้อง + เปิดหน้าต่าง
"""
import sys
import time

import cv2
from ultralytics import YOLO

weights = sys.argv[1] if len(sys.argv) > 1 else "best.onnx"
source = sys.argv[2] if len(sys.argv) > 2 else "0"
show = "--show" in sys.argv

model = YOLO(weights, task="detect")
cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
assert cap.isOpened(), f"cannot open source: {source}"

frames, t0 = 0, time.time()
while True:
    ok, frame = cap.read()
    if not ok:
        break

    result = model(frame, imgsz=320, verbose=False)[0]
    frames += 1

    if show:
        cv2.imshow("detect", result.plot())
        if cv2.waitKey(1) == 27:  # Esc
            break

cap.release()
elapsed = time.time() - t0
print(f"{frames} frames / {elapsed:.1f}s = {frames / elapsed:.1f} FPS")
