# reference: https://blog.francium.tech/feature-detection-and-matching-with-opencv-5fd2394a590
import numpy as np
import cv2
from matplotlib import pyplot as plt

# read image (uncomment the image you want to use)
img = cv2.imread('HG_06.jpg')
'''
img = cv2.imread('HG_no_grid_01.jpg') # uncomment to use image without the grid
'''
## This is the SHI-TOMASI Corner Method
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # makes the image into gray-scale

# Detect a maximum of 50 corners in the image
corners = cv2.goodFeaturesToTrack(gray_img, maxCorners=50, qualityLevel=0.02, minDistance=20) 
corners = np.float32(corners)

# show a green circle on the corner to show the keypoints in the image
for item in corners:
    x, y = item[0]
    cv2.circle(img, (x, y), 6, (0, 255, 0), -1)

# This is the ORB method
orb = cv2.ORB_create(nfeatures=2000) # detects 2000 feature points in the image
kp, des = orb.detectAndCompute(gray_img, None)

kp_img = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0) # draws the keypoinys as green points

# Save the Results (uncomment whether with grid or no grid)
'''
cv2.imwrite('ORB_grid.jpg', kp_img)
cv2.imwrite('shi__grid.jpg', kp_img)

cv2.imwrite('ORB_no_grid.jpg', kp_img)
cv2.imwrite('shi_tomasi_no_grid.jpg', img)
'''

# Show the Images
cv2.imshow('ORB', kp_img)
cv2.imshow('Shi-Tomasi', img)
cv2.waitKey(0)
