"""เทรน (ขั้น 5) — หา pretrained weight ให้เอง แล้วเลือก flag ให้ถูก

    python train.py           # epochs อัตโนมัติ
    python train.py 100       # กำหนด epochs เอง
"""
import sys
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import SETTINGS

spots = [Path("yolo11n.pt"), Path.home() / "yolo11n.pt", Path(SETTINGS["weights_dir"]) / "yolo11n.pt"]
pretrained = next((p for p in spots if p.exists()), None)

epochs = int(sys.argv[1]) if len(sys.argv) > 1 else (50 if pretrained else 200)
model = str(pretrained) if pretrained else "yolo11n.yaml"

print(f"model={model}  epochs={epochs}  {'' if pretrained else '<-- FROM SCRATCH (ไม่เจอ .pt)'}")

YOLO(model).train(
    data="data.yaml",
    epochs=epochs,
    imgsz=640,
    batch=16,
    # ponytail: AMP check ของ ultralytics ไปโหลด yolo11n.pt จากเน็ตเสมอ
    # เปิดได้เฉพาะตอนไฟล์นั้นมีอยู่จริง ไม่งั้นค้างรอเน็ต (เปิดแล้วเทรนเร็วขึ้น ~2x บน GPU)
    amp=Path("yolo11n.pt").exists(),
    plots=False,
)
