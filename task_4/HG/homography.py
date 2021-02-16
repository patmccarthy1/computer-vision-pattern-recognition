import cv2
import numpy as np
import matplotlib.pyplot as plt

# define images to be read in
img1_name = '../../images/HG_01.jpg'
img2_name = '../../images/HG_02.jpg'

# read in images
img1 = cv2.imread(img1_name, 0) 
img2 = cv2.imread(img2_name, 0)
plt.figure()
plt.imshow(img1)
plt.figure()
plt.imshow(img2)
# create SIFT object for keypoint detection and descriptor creation
orb = cv2.ORB_create()

# find keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object for descriptor matching
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# match descriptors
matches = bf.match(des1,des2)

# create variables for matched keypoints
image_1_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
image_2_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
for i in range(0,len(matches)):
    image_1_points[i] = kp1[matches[i].queryIdx].pt
    image_2_points[i] = kp2[matches[i].trainIdx].pt
H, _ = cv2.findHomography(image_1_points, image_2_points, cv2.RANSAC, ransacReprojThreshold=2.0)