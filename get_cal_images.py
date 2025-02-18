import cv2
import os

# Camera IDs (adjust if necessary)
LEFT_CAMERA_ID = 0
RIGHT_CAMERA_ID = 1

# Folder to save images
SAVE_FOLDER = "calibration_images"
os.makedirs(SAVE_FOLDER, exist_ok=True)

leftCam = cv2.VideoCapture(LEFT_CAMERA_ID)
rightCam = cv2.VideoCapture(RIGHT_CAMERA_ID)

frame_id = 0

while True:
    _, leftFrame = leftCam.read()
    _, rightFrame = rightCam.read()

    cv2.imshow("Left Camera", leftFrame)
    cv2.imshow("Right Camera", rightFrame)

    key = cv2.waitKey(1)
    if key == ord("s"):  # Press 's' to save a frame
        cv2.imwrite(f"{SAVE_FOLDER}/left_{frame_id}.png", leftFrame)
        cv2.imwrite(f"{SAVE_FOLDER}/right_{frame_id}.png", rightFrame)
        print(f"Saved frame {frame_id}")
        frame_id += 1

    elif key == ord("q"):  # Press 'q' to quit
        break

leftCam.release()
rightCam.release()
cv2.destroyAllWindows()
