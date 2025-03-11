import cv2
import zipfile
import os
import sys

def load_labels_from_zip(zip_path):
    """
    Extract label files from a zip archive and return as a dictionary.
    """
    labels = {}
    try:
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            print(f"Opened zip file {zip_path}")
            for file_name in zipf.namelist():
                if file_name.endswith('.txt'):
                    with zipf.open(file_name) as f:
                        frame_id = file_name.split('/')[-1].replace('.txt', '')  # Get frame ID without extension
                        labels[frame_id] = f.read().decode().splitlines()
                    print(f"Loaded labels for {frame_id}")
    except Exception as e:
        print(f"Error loading labels from zip: {e}")

    return labels

def preview_video_with_labels(video_path, zip_path):
    print(f"Starting preview for video: {video_path} and labels: {zip_path}")
    
    # Load labels from zip file
    labels = load_labels_from_zip(zip_path)
    print(f"Loaded {len(labels)} label files from zip.")
    
    # Open video
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print(f"Error: Unable to open video file {video_path}.")
        return

    frame_index = 0
    success, frame = video_capture.read()
    
    # Create a window only once
    cv2.namedWindow("Frame Preview", cv2.WINDOW_NORMAL)  # Create window once

    while success:
        print(f"Processing frame {frame_index:06d}...")
        
        # Check if there's a corresponding label for the frame
        frame_id = f"frame_{frame_index:06d}"
        if frame_id in labels:
            print(f"Found labels for {frame_id}.")
            # Draw bounding boxes on the frame
            for line in labels[frame_id]:
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Convert bounding box from normalized to pixel coordinates
                h, w, _ = frame.shape
                x_center_pixel = int(x_center * w)
                y_center_pixel = int(y_center * h)
                width_pixel = int(width * w)
                height_pixel = int(height * h)

                # Calculate top-left and bottom-right corner
                x1 = x_center_pixel - width_pixel // 2
                y1 = y_center_pixel - height_pixel // 2
                x2 = x_center_pixel + width_pixel // 2
                y2 = y_center_pixel + height_pixel // 2

                # Draw rectangle around the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                print(f"Drew bbox: {class_id} ({x1}, {y1}), ({x2}, {y2})")

        else:
            print(f"No labels found for frame {frame_id}.")

        # Display the frame with bounding boxes in the same window
        cv2.imshow("Frame Preview", frame)
        cv2.waitKey(0)
        
        # Wait for key press to move to next frame or quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Press 'q' to quit
            print("Quitting preview...")
            break
        
        frame_index += 1
        success, frame = video_capture.read()

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Get arguments
    if len(sys.argv) != 3:
        print("Usage: python3 preview.py <video_path> <zip_path>")
    else:
        video_path = sys.argv[1]
        zip_path = sys.argv[2]
        
        preview_video_with_labels(video_path, zip_path)
