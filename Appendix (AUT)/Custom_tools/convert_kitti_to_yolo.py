import os
from pathlib import Path

CLASSES = {
    'Car': 0,
    'Cyclist': 1,
    'Pedestrian': 2,
    'Van': 3,
    'Truck': 4,
    'Person_sitting': 5,
    'Tram': 6,
    'Misc': 7
}

def convert_kitti_to_yolo(kitti_label_path, yolo_label_path, img_width, img_height):
 
    with open(kitti_label_path, 'r') as f:
        lines = f.readlines()
    
    yolo_labels = []
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 15:
            continue
        
        class_name = parts[0]
        
        if class_name not in CLASSES:
            continue
        
        class_id = CLASSES[class_name]
        
        xmin = float(parts[4])
        ymin = float(parts[5])
        xmax = float(parts[6])
        ymax = float(parts[7])
        
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))
        
        yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    with open(yolo_label_path, 'w') as f:
        f.writelines(yolo_labels)

def convert_dataset(kitti_labels_dir, yolo_labels_dir, img_width=1242, img_height=375):

    os.makedirs(yolo_labels_dir, exist_ok=True)
    kitti_files = list(Path(kitti_labels_dir).glob('*.txt'))
    print(f"Konverterer {len(kitti_files)}")
    
    for kitti_file in kitti_files:
        yolo_file = Path(yolo_labels_dir) / kitti_file.name
        convert_kitti_to_yolo(str(kitti_file), str(yolo_file), img_width, img_height)
    
    print(f"Done: {yolo_labels_dir}")

if __name__ == "__main__":
    KITTI_LABELS = r"C:\Users\emilt\Desktop\dataset_aut - Kopi\val\labels_val_2"
    YOLO_LABELS = r"C:\Users\emilt\Desktop\dataset_aut - Kopi\val\Yolo_val_2"
    
    convert_dataset(KITTI_LABELS, YOLO_LABELS, img_width=1242, img_height=375)
    
