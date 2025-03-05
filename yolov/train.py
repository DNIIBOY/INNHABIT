import ultralytics
from ultralytics import YOLO
ultralytics.checks()
import shutil
import os
import cv2
import glob
import random
import zipfile
# Delete the dataset folder if it exists
if os.path.exists("dataset/"):
    shutil.rmtree("dataset/")
    print("✅ Dataset folder wiped successfully.")

# Recreate empty dataset structure
os.makedirs(f"dataset/images/train", exist_ok=True)
os.makedirs(f"dataset/images/val", exist_ok=True)
os.makedirs(f"dataset/labels/train", exist_ok=True)
os.makedirs(f"dataset/labels/val", exist_ok=True)

print("✅ Fresh dataset structure created!")
video_path = "Video.mp4"
output_folder = "dataset/images"
cap = cv2.VideoCapture(video_path)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_filename = os.path.join("dataset/images", f"frame_{frame_count:06d}.jpg")
    cv2.imwrite(frame_filename, frame)
    frame_count += 1

cap.release()
print(f"Extracted {frame_count} frames.")

zip_path = "labels.zip"
extract_to = "dataset/labels/"    # Target directory where files should go

# Ensure the target directory exists
os.makedirs(extract_to, exist_ok=True)

# Extract all files directly into /content/labels
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    for member in zip_ref.namelist():
        filename = os.path.basename(member)  # Get the actual filename
        if filename:  # Skip directories
            zip_ref.extract(member, extract_to)
            extracted_path = os.path.join(extract_to, member)
            new_path = os.path.join(extract_to, filename)
            os.rename(extracted_path, new_path)  # Move file to root

print(f"All files extracted directly to {extract_to}")
image_paths = glob.glob("dataset/images/*.jpg")
random.shuffle(image_paths)

split_idx = int(len(image_paths) * 0.8)
train_images = image_paths[:split_idx]
val_images = image_paths[split_idx:]

for img_path in train_images:
    shutil.move(img_path, "dataset/images/train/")
    label_path = img_path.replace("images", "labels").replace(".jpg", ".txt")
    if os.path.exists(label_path):
        shutil.move(label_path, "dataset/labels/train/")

for img_path in val_images:
    shutil.move(img_path, "dataset/images/val/")
    label_path = img_path.replace("images", "labels").replace(".jpg", ".txt")
    if os.path.exists(label_path):
        shutil.move(label_path, "dataset/labels/val/")

print("Dataset split completed!")

yaml_content = """train: images/train
val: images/val

nc: 1  # Change this based on the number of classes
names: ["object"]  # Change to actual class names
"""

with open("dataset/data.yaml", "w") as file:
    file.write(yaml_content)

print("data.yaml file created!")

model = YOLO("best.pt")
model.train(data="dataset/data.yaml", epochs=50, imgsz=640)
