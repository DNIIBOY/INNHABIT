import random
import cv2
import os
import zipfile
import sys
import numpy as np

def random_brightness_contrast(frame):
    """
    Apply random brightness and contrast to the frame.
    """
    alpha = random.uniform(0.6, 1.4)  # Random contrast factor
    beta = random.uniform(-40, 40)    # Random brightness factor
    
    # Apply the contrast and brightness
    frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    
    return frame

def random_saturation_hue(frame):
    """
    Apply random saturation and hue shift to the frame.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Convert to HSV space

    # Random saturation (0.5 to 1.5)
    saturation = random.uniform(0.5, 1.5)
    hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0, 255)

    # Random hue shift (-10 to 10 degrees)
    hue_shift = random.randint(-15, 15)
    
    # Convert hue to signed int (int16) to handle negative values
    hsv[..., 0] = hsv[..., 0].astype(np.int16)  # Convert to signed integer (int16)

    # Apply hue shift and wrap the value within the 0-179 range
    hsv[..., 0] = (hsv[..., 0].astype(np.int16) + hue_shift) % 180  # Hue is in range [0, 179]
    
    # Ensure the hue value stays within [0, 179]
    hsv[..., 0] = np.clip(hsv[..., 0], 0, 179)

    # Convert back to unsigned 8-bit (uint8)
    hsv[..., 0] = hsv[..., 0].astype(np.uint8)

    # Convert back to BGR
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return frame

def apply_random_color_transformations(frame):
    """
    Apply random color-related transformations to the frame.
    """
    # Randomly choose which transformation to apply
    transformations = [random_brightness_contrast, random_saturation_hue]
    chosen_transformation = random.choice(transformations)
    
    # Apply the chosen transformation
    frame = chosen_transformation(frame)
    
    return frame

def adjust_bounding_box_for_augmentation(bbox, image_shape, transformation):
    """
    Adjust bounding box coordinates based on the transformation applied to the image.
    The output bounding box is in normalized coordinates (0 to 1).
    """
    h, w, _ = image_shape
    x_center, y_center, width, height = bbox

    if transformation == "flip":
        # Flipping horizontally: invert x_center around the center of the image
        x_center = 1 - x_center  # x_center is normalized between 0 and 1
    elif transformation == "rotate":
        # Rotation logic to update coordinates can be added here (for future augmentations)
        pass  # Add more transformations here if needed.

    # Convert from normalized (0, 1) coordinates to pixel values
    x_center_pixel = int(x_center * w)
    y_center_pixel = int(y_center * h)
    width_pixel = int(width * w)
    height_pixel = int(height * h)

    # Make sure that the bounding box stays within image bounds
    x_center_pixel = max(0, min(x_center_pixel, w))
    y_center_pixel = max(0, min(y_center_pixel, h))
    width_pixel = max(0, min(width_pixel, w))
    height_pixel = max(0, min(height_pixel, h))

    # Convert the bounding box back to normalized coordinates
    x_center_norm = x_center_pixel / w
    y_center_norm = y_center_pixel / h
    width_norm = width_pixel / w
    height_norm = height_pixel / h

    # Return the adjusted bounding box in normalized coordinates (0 to 1)
    return x_center_norm, y_center_norm, width_norm, height_norm

def extract_zip(zip_path, extract_to_dir):
    """
    Extract all files from the zip archive into a given directory without nesting.
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # List all the files in the zip archive
        file_list = zip_ref.namelist()
        
        # Check if there's a single top-level directory in the zip
        top_level_dir = file_list[0].split('/')[0] if file_list else ''
        
        # If the files are inside a single folder, adjust the extract_to_dir path
        if top_level_dir:
            # Remove top-level directory from the extraction path
            for file_name in file_list:
                # If it's a directory, create it
                if file_name.endswith('/'):  # This indicates a directory
                    new_path = os.path.join(extract_to_dir, file_name)
                    os.makedirs(new_path, exist_ok=True)
                else:
                    # For files, extract and write them
                    new_path = os.path.join(extract_to_dir, os.path.relpath(file_name, top_level_dir))
                    os.makedirs(os.path.dirname(new_path), exist_ok=True)  # Create the parent directories
                    with open(new_path, 'wb') as f_out:
                        f_out.write(zip_ref.read(file_name))
            print(f"Extracted {zip_path} contents to {extract_to_dir} without nesting.")
        else:
            # If no top-level folder exists, just extract to the provided directory
            zip_ref.extractall(extract_to_dir)
            print(f"Extracted {zip_path} to {extract_to_dir}")

def augment_and_create_video_from_zip_mp4(video_path, zip_path, output_name):
    print(f"Starting augmentation process...")
    print(f"Video Path: {video_path}, Zip Path: {zip_path}, Output Name: {output_name}")
    
    # Create output directories
    augmented_images_dir = f"{output_name}_images"
    os.makedirs(augmented_images_dir, exist_ok=True)

    # Open the video and extract frames as usual
    video_capture = cv2.VideoCapture(video_path)
    
    if not video_capture.isOpened():
        print(f"Error: Unable to open video file {video_path}.")
        return
    
    success, frame = video_capture.read()
    frame_index = 0

    # Check if zip_path exists
    if not os.path.exists(zip_path):
        print(f"Error: The zip folder {zip_path} does not exist.")
        return

    # Extract the zip file content into a temporary folder
    extracted_folder = zip_path.rstrip('.zip')  # Remove the .zip extension for folder name
    extract_zip(zip_path, extracted_folder)

    # Video writer to save the augmented video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    augmented_video_path = f"{output_name}.mp4"
    out_video = cv2.VideoWriter(augmented_video_path, fourcc, 30.0, (frame.shape[1], frame.shape[0]))  # Adjust fps as needed

    # Create a list to store labels
    label_files = []

    # Assuming the label data is in the extracted folder, we'll process each frame
    while success:
        print(f"Processing frame {frame_index:06d}...")
        
        # Example of augmentation: Random horizontal flip
        if random.random() < 0.5:
            frame = cv2.flip(frame, 1)  # Horizontal flip
            transformation = "flip"
        else:
            transformation = "none"
        frame = apply_random_color_transformations(frame)
        # Check if the corresponding label file exists in the extracted folder
        label_file = f"{extracted_folder}/frame_{frame_index:06d}.txt"
        print(f"Looking for label file: {label_file}")
        
        if os.path.exists(label_file):
            with open(label_file, "r") as f:
                lines = f.readlines()

            new_bboxes = []
            for line in lines:
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                # Adjust bounding box for the current transformation
                new_bbox = adjust_bounding_box_for_augmentation((x_center, y_center, width, height), frame.shape, transformation)
                new_bboxes.append(f"{class_id} {new_bbox[0]} {new_bbox[1]} {new_bbox[2]} {new_bbox[3]}")

                # Debugging: Print bounding box values after augmentation
                print(f"Original bbox: {x_center, y_center, width, height}")
                print(f"Augmented bbox for frame_{frame_index:06d}: {new_bbox}")
            
            # Debugging: Check if new_bboxes contains data
            if new_bboxes:
                print(f"Bounding boxes for frame_{frame_index:06d}: {new_bboxes}")
            else:
                print(f"No bounding boxes after augmentation for frame_{frame_index:06d}")

            # Save the updated label for the augmented image
            augmented_label_file = f"{augmented_images_dir}/frame_{frame_index:06d}.txt"
            with open(augmented_label_file, "w") as f:
                f.write("\n".join(new_bboxes))

            # Add label file to the list for zipping later
            label_files.append(augmented_label_file)
        else:
            print(f"Label file for frame_{frame_index:06d} not found!")

        # Save the augmented image
        augmented_image_file = f"{augmented_images_dir}/frame_{frame_index:06d}.jpg"
        cv2.imwrite(augmented_image_file, frame)

        # Write the augmented frame to the output video
        out_video.write(frame)

        frame_index += 1
        success, frame = video_capture.read()

    video_capture.release()
    out_video.release()

    # Debugging: Check if label files are collected
    print(f"Total label files: {len(label_files)}")

    # Create a zip file for the labels
    zip_file_name = f"{output_name}.zip"
    if label_files:
        with zipfile.ZipFile(zip_file_name, 'w') as zipf:
            for label_file in label_files:
                print(f"Adding {label_file} to zip")
                zipf.write(label_file, os.path.basename(label_file))  # Add label file to zip with original name
        print(f"Labels saved in {zip_file_name}.")
    else:
        print("No labels to save. The zip file will be empty.")

    print(f"Augmented video saved as {augmented_video_path}.")

if __name__ == "__main__":
    # Get arguments
    if len(sys.argv) != 4:
        print("Usage: python3 augment.py <video_path> <zip_path> <output_name>")
    else:
        video_path = sys.argv[1]
        zip_path = sys.argv[2]
        output_name = sys.argv[3]
        
        augment_and_create_video_from_zip_mp4(video_path, zip_path, output_name)
