# reference: https://github.com/jagracar/OpenCV-python-tests/blob/master/OpenCV-tutorials/cameraCalibration/depthMap.py
import numpy as np
import cv2
from matplotlib import pyplot as plt

imgL = cv2.imread('rectified_6.jpg',0)
imgR = cv2.imread('rectified_5.jpg',0)

# Initialize the stereo block matching object 
stereo = cv2.StereoBM_create(numDisparities=32, blockSize=13)

# Compute the disparity image
disparity = stereo.compute(imgL, imgR)

# Normalize the image for representation
min = disparity.min()
max = disparity.max()
disparity = np.uint8(255 * (disparity - min) / (max - min))

# Display the result
cv2.imshow('disparity', np.hstack((imgL, imgR, disparity)))
cv2.waitKey(0)
cv2.imwrite('results/depth_map3.jpg',np.hstack((imgL, imgR, disparity)))
cv2.destroyAllWindows()

