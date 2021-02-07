import numpy as np
import cv2
from matplotlib import pyplot as plt

# read image
orgS = cv2.imread('FD_01.jpg')
# Resize image
img = cv2.resize(orgS, (400, 450))  

## This is the SHI-TOMASI Corner Method
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray_img, maxCorners=50, qualityLevel=0.02, minDistance=20)
corners = np.float32(corners)

for item in corners:
    x, y = item[0]
    cv2.circle(img, (x, y), 6, (0, 255, 0), -1)

# This is the ORB method
orb = cv2.ORB_create(nfeatures=2000)
kp, des = orb.detectAndCompute(gray_img, None)

kp_img = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)

cv2.imshow('ORB', kp_img)
cv2.imshow('Shi-Tomasi', img)
cv2.waitKey(0)
