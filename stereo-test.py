import numpy as np
import cv2
from matplotlib import pyplot as plt
import time

imgLeft = cv2.imread('scene11.png', 0)
imgRight = cv2.imread('scene12.png', 0)
def main():
    start_time = time.time()
    result = ShowDisparity(bSize=15)
    print(time.time() - start_time)


def ShowDisparity(bSize=15):
    # Initialize the stereo block matching object 
    stereo = cv2.StereoBM_create(numDisparities=32, blockSize=bSize)

    # Compute the disparity image
    disparity = stereo.compute(imgLeft, imgRight)

    # Normalize the image for representation
    min = disparity.min()
    max = disparity.max()
    disparity = np.uint8(255 * (disparity - min) / (max - min))
    
    # Plot the result
    return disparity



if __name__ == "__main__":
    main()
