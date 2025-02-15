import numpy as np
import cv2
from matplotlib import pyplot as plt
import time

print(cv2.ocl.haveOpenCL())

cv2.ocl.setUseOpenCL(True)
cam_left = cv2.VideoCapture(4)
cam_right = cv2.VideoCapture(6)


def main():
    while True:
        start_time = time.time()
        result = ShowDisparity(bSize=55)
        print(time.time() - start_time)
        print(1 / (time.time() - start_time))
        cv2.imshow("test", result)
        cv2.waitKey(int (1000 * (time.time() - start_time)))
def ShowDisparity(bSize=15):
    retl, frame_left = cam_left.read()
    retr, frame_right = cam_right.read()
    frame_left_gray = cv2.UMat(cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY))
    frame_right_gray = cv2.UMat(cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY))
    stereo = cv2.StereoBM_create(numDisparities=32, blockSize=bSize)
    disparity = stereo.compute(frame_left_gray, frame_right_gray).get()

    min = disparity.min()
    max = disparity.max()
    disparity = np.uint8(255 * (disparity - min) / (max - min))

    return disparity



if __name__ == "__main__":
    main()
