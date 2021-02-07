import numpy as np
import cv2
from matplotlib import pyplot as plt

img_org= cv2.imread('FD_01.jpg', 0)
img1 = cv2.resize(img_org, (400, 450)) 

img_twist = cv2.imread('', 0)
img2 = cv2.resize(img_twist, (400, 450))

orb = cv2.ORB_create(nfeatures=500)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# matcher takes normType, which is set to cv2.NORM_L2 for SIFT and SURF, cv2.NORM_HAMMING for ORB, FAST and BRIEF
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
# draw first 50 matches
match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None)

cv2.imshow('Matches', match_img)
cv2.waitKey(0)
