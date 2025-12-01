from pathlib import Path

IMAGES_DIR = Path(r"C:\Users\emilt\Desktop\DTU\Semester 2\34759 Perception for Autonomous Systems\Final Project\train_yolo_55\training\image")
LABELS_DIR = Path(r"C:\Users\emilt\Desktop\DTU\Semester 2\34759 Perception for Autonomous Systems\Final Project\train_yolo_55\training\labels")

IMG_EXTS = [".jpg", ".jpeg", ".png"]
DRY_RUN = False  

def main():
    if not IMAGES_DIR.exists() or not LABELS_DIR.exists():
        print(f"IMAGES_DIR found: {IMAGES_DIR.exists()}")
        print(f"LABELS_DIR found: {LABELS_DIR.exists()}")
        return

    images = [p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in IMG_EXTS]
    print(f"Found {len(images)} pic.\n")

    removed = 0
    kept = 0

    for img in images:
        stem = img.stem
        label_path = LABELS_DIR / f"{stem}.txt"
        if not label_path.exists():
            if DRY_RUN:
                print(f"[DRY_RUN] Gonna be deleted: {img}")
            else:
                img.unlink()
                print(f"Deleted: {img}")
            removed += 1
        else:
            kept += 1

    print("\n=== RESUMÉ ===")
    print(f"Kept: {kept}")
    print(f"Deleted: {removed}")
    if DRY_RUN:
        print("\nDRY_RUN")

if __name__ == "__main__":
    main()