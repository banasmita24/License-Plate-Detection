import os
import cv2
import shutil
from tqdm import tqdm

# Root directory of CCPD2019 dataset
dataset_root = "CCPD2019"
splits_folder = os.path.join(dataset_root, "splits")

# Output YOLOv8-style dataset folder
output_root = "ccpd_yolov8"
os.makedirs(output_root, exist_ok=True)

# Function to ensure folder exists
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Loop through train, val, test
for split in ["train", "val", "test"]:
    split_file = os.path.join(splits_folder, f"{split}.txt")
    if not os.path.exists(split_file):
        print(f"⚠️ Missing split file: {split_file}")
        continue

    with open(split_file, "r") as f:
        image_paths = [line.strip() for line in f.readlines()]

    image_output_dir = os.path.join(output_root, "images", split)
    label_output_dir = os.path.join(output_root, "labels", split)
    ensure_dir(image_output_dir)
    ensure_dir(label_output_dir)

    for rel_image_path in tqdm(image_paths, desc=f"Processing {split}"):
        img_path = os.path.join(dataset_root, rel_image_path)
        if not os.path.exists(img_path):
            print(f"❌ Image not found: {img_path}")
            continue

        filename = os.path.basename(img_path)
        name_no_ext = os.path.splitext(filename)[0]

        try:
            # Bounding box parsing
            fields = name_no_ext.split("-")
            bbox_str = fields[2]
            top_left, bottom_right = bbox_str.split("_")
            x1, y1 = map(int, top_left.split("&"))
            x2, y2 = map(int, bottom_right.split("&"))
        except Exception as e:
            print(f"❌ Error parsing {filename}: {e}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Failed to read image: {img_path}")
            continue

        h, w = img.shape[:2]

        # YOLOv8 normalized format: class_id x_center y_center width height
        x_center = ((x1 + x2) / 2.0) / w
        y_center = ((y1 + y2) / 2.0) / h
        bbox_width = abs(x2 - x1) / w
        bbox_height = abs(y2 - y1) / h

        yolo_label = f"0 {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n"

        # Save label
        label_file_path = os.path.join(label_output_dir, name_no_ext + ".txt")
        with open(label_file_path, "w") as f:
            f.write(yolo_label)

        # Copy image to YOLOv8 folder
        dst_img_path = os.path.join(image_output_dir, filename)
        shutil.copyfile(img_path, dst_img_path)

print("✅ Done! YOLOv8 dataset is ready in the 'ccpd_yolov8' folder.")
