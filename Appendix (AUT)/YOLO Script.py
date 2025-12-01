from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
from ultralytics import YOLO


def draw_and_print_results(
    img,
    results,
    model: YOLO,
    color: Tuple[int, int, int],
    start_index: int,
) -> int:
    if results.boxes is None or results.boxes.xyxy is None:
        return start_index

    current_id = start_index

    for box, cls, conf in zip(
        results.boxes.xyxy,
        results.boxes.cls,
        results.boxes.conf,
    ):
        cls_idx = int(cls)

        if cls_idx < 0 or cls_idx >= len(model.names):
            continue

        class_name = str(model.names[cls_idx])
        if not class_name or class_name.strip() == "":
            continue

        x1, y1, x2, y2 = map(int, box)
        width = x2 - x1
        height = y2 - y1
        area = width * height
        cx = x1 + width / 2.0
        cy = y1 + height / 2.0

        label = f"{class_name} {conf:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img,
            label,
            (x1, max(y1 - 10, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

        cv2.putText(
            img,
            str(current_id),
            (int(cx), int(cy)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

        print(
            f"  Detection {current_id}: "
            f"class={class_name}, conf={conf:.2f}, "
            f"x1={x1}, y1={y1}, x2={x2}, y2={y2}, "
            f"w={width}, h={height}, area={area}, "
            f"center=({cx:.1f}, {cy:.1f})"
        )

        current_id += 1

    return current_id


## Changes this to the right path where your images are stored
def main() -> None:
    images_dir = Path(
        r"C:\Users\emilt\Desktop\DTU\Semester 2\34759 Perception for Autonomous Systems"
        r"\Final Project\train_yolo_55\test\image"
    )

    script_dir = Path(__file__).resolve().parent
    model_files = sorted(script_dir.glob("*.pt"))
    if not model_files:
        print(f"No .pt files found in {script_dir}")
        return

    models = [YOLO(str(p)) for p in model_files]

    image_paths = sorted(
        list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
    )

    if not image_paths:
        print(f"No images found in {images_dir}")
        return

    print(f"Found {len(image_paths)} images in {images_dir}")

    red = (0, 0, 255)

    for img_path in image_paths:
        print("\n" + "=" * 80)
        print(f"Image: {img_path.name}")

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Could not read: {img_path}")
            continue

        img_out = img.copy()
        current_id = 1

        for model in models:
            results = model(img, conf=0.4, verbose=False)[0]
            current_id = draw_and_print_results(
                img_out,
                results,
                model,
                red,
                current_id,
            )

        cv2.imshow("Detections", img_out)

        key = cv2.waitKey(1000) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    print("Finished processing all images.")


if __name__ == "__main__":
    main()
