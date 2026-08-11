"""แบ่ง train/val หลัง export YOLO จาก Label Studio

    python split.py dataset

ก่อนรัน:  dataset/images/*.jpg + dataset/labels/*.txt
หลังรัน:  dataset/images/train  dataset/images/val
          dataset/labels/train  dataset/labels/val
"""
import random
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "dataset")

imgs = sorted(p for p in (root / "images").glob("*.*") if p.is_file())
assert imgs, f"no images in {root / 'images'} (already split, or wrong path)"

random.Random(0).shuffle(imgs)
n_val = max(1, len(imgs) // 5)  # 20%

for i, img in enumerate(imgs):
    part = "val" if i < n_val else "train"
    lbl = root / "labels" / f"{img.stem}.txt"
    for src, sub in ((img, "images"), (lbl, "labels")):
        dst = root / sub / part
        dst.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.move(str(src), str(dst / src.name))
        else:
            (dst / src.name).touch()  # รูปที่ไม่มี object ต้องมี .txt ว่างคู่ไว้

print(f"train {len(imgs) - n_val} imgs / val {n_val} imgs")
