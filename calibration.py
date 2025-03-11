import glob

import cv2
import numpy as np

# Chessboard properties
CHESSBOARD_SIZE = (9, 6)  # Adjust to match your chessboard
SQUARE_SIZE = 2.5  # Size of a square in cm (adjust as needed)

# Termination criteria for corner refinement
TERMINATION_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (like a 3D chessboard)
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = (
    np.mgrid[0 : CHESSBOARD_SIZE[0], 0 : CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    * SQUARE_SIZE
)

# Arrays to store detected corners and corresponding 3D points
obj_points = []  # 3D points in real-world space
left_img_points = []  # 2D points in left images
right_img_points = []  # 2D points in right images

# Load captured images
left_images = sorted(glob.glob("calibration_images/left_*.png"))
right_images = sorted(glob.glob("calibration_images/right_*.png"))

assert len(left_images) == len(right_images), "Mismatched image pairs!"

for left_img_path, right_img_path in zip(left_images, right_images):
    left_img = cv2.imread(left_img_path)
    right_img = cv2.imread(right_img_path)

    gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    found_left, corners_left = cv2.findChessboardCorners(
        gray_left, CHESSBOARD_SIZE, None
    )
    found_right, corners_right = cv2.findChessboardCorners(
        gray_right, CHESSBOARD_SIZE, None
    )

    if found_left and found_right:
        obj_points.append(objp)

        corners_left = cv2.cornerSubPix(
            gray_left, corners_left, (11, 11), (-1, -1), TERMINATION_CRITERIA
        )
        corners_right = cv2.cornerSubPix(
            gray_right, corners_right, (11, 11), (-1, -1), TERMINATION_CRITERIA
        )

        left_img_points.append(corners_left)
        right_img_points.append(corners_right)

        # Draw and display
        cv2.drawChessboardCorners(left_img, CHESSBOARD_SIZE, corners_left, found_left)
        cv2.drawChessboardCorners(
            right_img, CHESSBOARD_SIZE, corners_right, found_right
        )
        cv2.imshow("Left", left_img)
        cv2.imshow("Right", right_img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Camera calibration
_, left_camera_matrix, left_dist_coeffs, _, _ = cv2.calibrateCamera(
    obj_points, left_img_points, gray_left.shape[::-1], None, None
)
_, right_camera_matrix, right_dist_coeffs, _, _ = cv2.calibrateCamera(
    obj_points, right_img_points, gray_right.shape[::-1], None, None
)

# Stereo calibration
(_, _, _, _, _, rotation_matrix, translation_vector, _, _) = cv2.stereoCalibrate(
    obj_points,
    left_img_points,
    right_img_points,
    left_camera_matrix,
    left_dist_coeffs,
    right_camera_matrix,
    right_dist_coeffs,
    gray_left.shape[::-1],
    None,
    None,
    None,
    None,
    cv2.CALIB_FIX_INTRINSIC,
    TERMINATION_CRITERIA,
)

# Stereo rectification
(
    left_rectification,
    right_rectification,
    left_projection,
    right_projection,
    disparity_to_depth_map,
    left_roi,
    right_roi,
) = cv2.stereoRectify(
    left_camera_matrix,
    left_dist_coeffs,
    right_camera_matrix,
    right_dist_coeffs,
    gray_left.shape[::-1],
    rotation_matrix,
    translation_vector,
    None,
    None,
    None,
    None,
    None,
    cv2.CALIB_ZERO_DISPARITY,
    alpha=-1,
)

# Compute rectification maps
left_map_x, left_map_y = cv2.initUndistortRectifyMap(
    left_camera_matrix,
    left_dist_coeffs,
    left_rectification,
    left_projection,
    gray_left.shape[::-1],
    cv2.CV_32FC1,
)
right_map_x, right_map_y = cv2.initUndistortRectifyMap(
    right_camera_matrix,
    right_dist_coeffs,
    right_rectification,
    right_projection,
    gray_right.shape[::-1],
    cv2.CV_32FC1,
)

# Save calibration data
np.savez(
    "calibration_data.npz",
    imageSize=gray_left.shape[::-1],
    leftCameraMatrix=left_camera_matrix,
    leftDistortion=left_dist_coeffs,
    rightCameraMatrix=right_camera_matrix,
    rightDistortion=right_dist_coeffs,
    rotationMatrix=rotation_matrix,
    translationVector=translation_vector,
    leftMapX=left_map_x,
    leftMapY=left_map_y,
    rightMapX=right_map_x,
    rightMapY=right_map_y,
    leftROI=left_roi,
    rightROI=right_roi,
)

print("Calibration complete! Saved to 'calibration_data.npz'.")
