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
# Function to extract frames from video (unchanged)
def extract_frames_from_videos(video_dir, output_base):
    os.makedirs(output_base, exist_ok=True)  # Ensure the output base directory exists
    
    video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
    
    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        video_name = os.path.splitext(video_file)[0]  # Remove .mp4 extension
        output_folder = os.path.join(output_base, video_name)
        
        os.makedirs(output_folder, exist_ok=True)  # Create folder for this video
        
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_filename = os.path.join(output_folder, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_count += 1
        
        cap.release()
        print(f"Extracted {frame_count} frames from {video_file}.")

# Extract labels from zip files (unchanged)
def extract_labels(zip_dir, output_label_dir):
    os.makedirs(output_label_dir, exist_ok=True)
    
    zip_files = glob.glob(f"{zip_dir}/*.zip")
    for zip_path in zip_files:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_label_dir)  # Extract directly into labels root
        print(f"Extracted {zip_path} to {output_label_dir}")

# Match frames to labels and ensure they are correctly placed for YOLO
def organize_labels_and_frames(image_dir, label_dir, output_image_dir, output_label_dir):
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    # Traverse each subfolder (video folder) in the image directory
    video_folders = [f for f in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, f))]
    
    # Shuffle the video folders (not the individual files yet)
    random.shuffle(video_folders)
    
    # Function to rename and move files
    def process_video(video_folder, output_image_dir, output_label_dir, frame_index):
        video_image_dir = os.path.join(image_dir, video_folder)
        video_label_dir = os.path.join(label_dir, video_folder)

        # Get all image paths in this video folder
        image_paths = sorted(glob.glob(f"{video_image_dir}/*.jpg"))
        total_frames = len(image_paths)
        
        # Split the images into 80% train and 20% val
        split_idx = int(total_frames * 0.8)
        train_images = image_paths[:split_idx]
        val_images = image_paths[split_idx:]

        # Process training images and labels
        for img_path in train_images:
            new_frame_name = f"frame_{frame_index:06d}.jpg"
            new_img_path = os.path.join(output_image_dir, "train", new_frame_name)

            try:
                shutil.move(img_path, new_img_path)
                frame_index += 1  # Increment global frame index
            except Exception as e:
                print(f"Error moving image {img_path}: {e}")
                continue

            # Process and move the label file if it exists
            label_path = os.path.join(video_label_dir, os.path.basename(img_path).replace(".jpg", ".txt"))
            if os.path.exists(label_path):
                new_label_name = new_frame_name.replace(".jpg", ".txt")
                new_label_path = os.path.join(output_label_dir, "train", new_label_name)
                try:
                    shutil.move(label_path, new_label_path)
                except Exception as e:
                    print(f"Error moving label {label_path}: {e}")
            else:
                print(f"Warning: Label for {img_path} not found.")

        # Process validation images and labels
        for img_path in val_images:
            new_frame_name = f"frame_{frame_index:06d}.jpg"
            new_img_path = os.path.join(output_image_dir, "val", new_frame_name)

            try:
                shutil.move(img_path, new_img_path)
                frame_index += 1  # Increment global frame index
            except Exception as e:
                print(f"Error moving image {img_path}: {e}")
                continue

            # Process and move the label file if it exists
            label_path = os.path.join(video_label_dir, os.path.basename(img_path).replace(".jpg", ".txt"))
            if os.path.exists(label_path):
                new_label_name = new_frame_name.replace(".jpg", ".txt")
                new_label_path = os.path.join(output_label_dir, "val", new_label_name)
                try:
                    shutil.move(label_path, new_label_path)
                except Exception as e:
                    print(f"Error moving label {label_path}: {e}")
            else:
                print(f"Warning: Label for {img_path} not found.")
        
        return frame_index  # Return the updated frame index

    # Process each video folder
    global_frame_index = 0
    for video_folder in video_folders:
        global_frame_index = process_video(video_folder, output_image_dir, output_label_dir, global_frame_index)

    print("Dataset split and labels merged successfully!")

    # Function to count and display the number of files in each folder
    def display_file_counts():
        # Counting images and labels for train/val
        for split_folder in ['train', 'val']:
            image_count = len(glob.glob(os.path.join(output_image_dir, split_folder, "*.jpg")))
            label_count = len(glob.glob(os.path.join(output_label_dir, split_folder, "*.txt")))
            print(f"{split_folder.capitalize()} set - Images: {image_count}, Labels: {label_count}")

    # Display file counts
    display_file_counts()
# Main code execution
video_directory = "."  # Path to video folder
output_image_directory = "dataset/images"  # Base output image folder
label_directory = "dataset/labels"  # Base output label folder

# Extract frames from videos and labels from zip files
extract_frames_from_videos(video_directory, output_image_directory)

label_root = "dataset/labels/"  # Label extraction directory
extract_labels(".", label_root)  # Assuming labels are stored in zip files in /content

# Organize the images and labels into train/val splits
organize_labels_and_frames(output_image_directory, label_root, output_image_directory, label_root)

print("Finished processing frames and labels.")
base_dir = os.path.abspath("dataset")  # Adjust this as needed, e.g., to the full path of your dataset directory

# Define the full paths for train and val directories
train_path = os.path.join(base_dir, "images", "train")
val_path = os.path.join(base_dir, "images", "val")

# YAML content with full paths
yaml_content = f"""train: {train_path}
val: {val_path}

nc: 1  # Change this based on the number of classes
names: ["object"]  # Change to actual class names
"""

# Write the YAML content to a file
with open("dataset/data.yaml", "w") as file:
    file.write(yaml_content)

print("data.yaml file created with full paths!")

print("data.yaml file created!")
def load_model():
    # Check if 'best.pt' exists in the specified directory
    best_model_path = "runs/detect/train/weights/best.pt"
    
    if os.path.exists(best_model_path):
        print(f"Found {best_model_path}, loading model...")
        model = YOLO(best_model_path)
        return model
    
    # If 'best.pt' does not exist, look for any .pt file in the /content folder
    pt_files = list(os.listdir("."))
    pt_files = [f for f in pt_files if f.endswith(".pt")]
    
    if pt_files:
        print(f"Found a .pt file: {pt_files[0]}, loading model...")
        model = YOLO(os.path.join(".", pt_files[0]))  # Load the first .pt file found
        return model
    
    # If no .pt file is found, fallback to running YOLOv8 with default model
    print("No model file found, using default YOLOv8 model and running training...")
    return YOLO("yolov8n.pt")  # Default YOLOv8 model, adjust if necessary

def train_model():
    model = load_model()  # Load the model (either best.pt, a fallback pt file, or the default YOLOv8 model)
    
    # Continue training or start fresh (you can adjust the params as necessary)
    model.train(data="dataset/data.yaml", epochs=50, imgsz=640, patience=20)

    # Run validation after training
    metrics = model.val()
    print(metrics)  # Outputs key evaluation metrics (e.g., mAP, Precision, Recall)

train_model()
