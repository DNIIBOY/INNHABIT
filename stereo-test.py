import numpy as np
import cv2
from matplotlib import pyplot as plt
import time


cam_left = cv2.VideoCapture(0, cv2.CAP_V4L2)
cam_right = cv2.VideoCapture(1, cv2.CAP_V4L2)
def main():
    while True:
        start_time = time.time()
        result = ShowDisparity(bSize=15)
        # print("Time", time.time() - start_time, "ms")
        print("FPS ", 1 / (time.time() - start_time))
        # cv2.imshow("test", result)
        # cv2.waitKey(int (1000 * (time.time() - start_time)))
def ShowDisparity(bSize=15):
    retl, frame_left = cam_left.read()
    retr, frame_right = cam_right.read()
    gpu_image_left = cv2.cuda.GpuMat()
    gpu_image_left.upload(frame_left)
    gpu_image_right = cv2.cuda.GpuMat()
    gpu_image_right.upload(frame_right)
    gpu_frame_left_gray = cv2.cuda.cvtColor(gpu_image_left, cv2.COLOR_BGR2GRAY)
    gpu_frame_right_gray = cv2.cuda.cvtColor(gpu_image_right, cv2.COLOR_BGR2GRAY)
    stereo = cv2.cuda.createStereoBM(numDisparities=16, blockSize=bSize)
    gpu_stream = cv2.cuda.Stream()
    disparity_gpu = stereo.compute(gpu_frame_left_gray, gpu_frame_right_gray, gpu_stream)
    disparity = disparity_gpu.download()
    min = disparity.min()
    max = disparity.max()
    disparity = np.uint8(255 * (disparity - min) / (max - min))

    return disparity



if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
