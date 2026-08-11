import cv2
from pathlib import Path
import random

INPUT_DIR = Path("dataset/images")
OUTPUT_DIR = Path("dataset_rotate")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for image_path in INPUT_DIR.glob("*"):

    if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    image = cv2.imread(str(image_path))

    if image is None:
        continue

    # สุ่มมุม -10 ถึง 10 องศา
    angle = random.uniform(-10, 10)

    h, w = image.shape[:2]

    center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    rotated = cv2.warpAffine(
        image,
        matrix,
        (w, h)
    )

    output_path = OUTPUT_DIR / image_path.name

    cv2.imwrite(
        str(output_path),
        rotated
    )

print("หมุนภาพเสร็จแล้ว")
