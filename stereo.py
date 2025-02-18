import cv2
import numpy as np
import sys

# Constants
LEFT_CAMERA_ID = 0  # Adjust if needed
RIGHT_CAMERA_ID = 1  # Adjust if needed
REMAP_INTERPOLATION = cv2.INTER_LINEAR

# Load calibration data
calibration = np.load("calibration_data.npz", allow_pickle=False)

imageSize = tuple(calibration["imageSize"])
leftMapX = calibration["leftMapX"]
leftMapY = calibration["leftMapY"]
rightMapX = calibration["rightMapX"]
rightMapY = calibration["rightMapY"]

# Initialize cameras
leftCam = cv2.VideoCapture(LEFT_CAMERA_ID)
rightCam = cv2.VideoCapture(RIGHT_CAMERA_ID)

if not (leftCam.isOpened() and rightCam.isOpened()):
    print("Error: Could not open cameras.")
    sys.exit()

# Create stereo matcher
stereoMatcher = cv2.StereoBM_create(numDisparities=16*5, blockSize=15)

while True:
    # Capture frames
    if not (leftCam.grab() and rightCam.grab()):
        print("No more frames")
        break

    _, leftFrame = leftCam.retrieve()
    _, rightFrame = rightCam.retrieve()

    # Rectify images
    fixedLeft = cv2.remap(leftFrame, leftMapX, leftMapY, REMAP_INTERPOLATION)
    fixedRight = cv2.remap(rightFrame, rightMapX, rightMapY, REMAP_INTERPOLATION)

    # Convert to grayscale
    grayLeft = cv2.cvtColor(fixedLeft, cv2.COLOR_BGR2GRAY)
    grayRight = cv2.cvtColor(fixedRight, cv2.COLOR_BGR2GRAY)

    # Compute disparity (depth map)
    depth = stereoMatcher.compute(grayLeft, grayRight)

    # Normalize disparity for visualization
    depth_display = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Show images
    cv2.imshow("Left Camera", fixedLeft)
    cv2.imshow("Right Camera", fixedRight)
    cv2.imshow("Depth Map", depth_display)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
leftCam.release()
rightCam.release()
cv2.destroyAllWindows()
