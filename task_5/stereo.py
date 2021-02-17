import cv2
import numpy as np
import matplotlib.pyplot as plt

# define images to be read in
img1_name = '../images/FD_03.jpg'
img2_name = '../images/FD_04.jpg'

# read in images 
img1 = cv2.imread(img1_name, 0) 
img2 = cv2.imread(img2_name, 0)

# create SIFT object for keypoint detection and descriptor creation
sift = cv2.SIFT_create()

# find keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# parameters for FLANN feature matching of keypoints
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)
pts1 = []
pts2 = []

# Lowe's ratio test for matching keypoints
for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)
     

# find the fundamental matrix
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
FDmatrix = np.array(F) 

# =============================================================================
# stereo rectification - uncalibrated
# =============================================================================

# calculate reprojection matrices for stereo rectification
h1, w1 = img1.shape
h2, w2 = img2.shape
_, H1, H2 = cv2.stereoRectifyUncalibrated(
    np.float32(pts1), np.float32(pts2), F, imgSize=(w1, h1)
)


# rectify images
img1_rectified = cv2.warpPerspective(img1, H1, (w1, h1))
img2_rectified = cv2.warpPerspective(img2, H2, (w2, h2))
cv2.imwrite("rectified-1.jpg", img1_rectified)
cv2.imwrite("rectified-2.jpg", img2_rectified)

# plot rectified images
fig, axes = plt.subplots(1, 2, figsize=(15, 10))
axes[0].imshow(img1_rectified, cmap="gray")
axes[1].imshow(img2_rectified, cmap="gray")
axes[0].axhline(250)
axes[1].axhline(250)
axes[0].axhline(450)
axes[1].axhline(450)
axes[0].get_xaxis().set_visible(False)
axes[0].get_yaxis().set_visible(False)
axes[1].get_xaxis().set_visible(False)
axes[1].get_yaxis().set_visible(False)
plt.savefig("rectified_images.png")
plt.show()

# =============================================================================
# stereo rectification - calibrated
# =============================================================================
# camera calibration parameters to ensure accurate stereo rectification 
# (see calibration.py in task_3 for how these were obtained)

## camera matrix
#K = [[404.7797,0.0000,265.5773],
#     [0.0000,417.2593,233.3487],
#     [0.0000,0.0000,1.0000]]
#K = np.array(K)
#
## distortion coefficients 
#d = [0.4885,-3.0033,0.0120,0.0203,4.3558] 
#d = np.array(d)
#
## rotation vector (extrinsic parameters):
#R = [0.4246,0.4581,0.2170]  # set to rotation vector for corresponding image
#R = np.array(R)
#
## translation vectors (extrinsic parameters) 
#T = [-1.4532,-3.0516,13.7285] # set to translation vector for corresponding image
#T = np.array(T)
#
#R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K,d,K,d,img1.shape[:2],R,T,alpha=1)
#mapx1, mapy1 = cv2.initUndistortRectifyMap(K,d,R1,K,img1.shape[:2],cv2.CV_32F)
#mapx2, mapy2 = cv2.initUndistortRectifyMap(K,d,R2,K,img2.shape[:2],cv2.CV_32F)
#img_rect1 = cv2.remap(img1, mapx1, mapy1, cv2.INTER_LINEAR)
#img_rect2 = cv2.remap(img2, mapx2, mapy2, cv2.INTER_LINEAR)
# 
## draw the images side by side
#total_size = (max(img_rect1.shape[0], img_rect2.shape[0]),
#              img_rect1.shape[1] + img_rect2.shape[1])
#img = np.zeros(total_size, dtype=np.uint8)
#img[:img_rect1.shape[0], :img_rect1.shape[1]] = img_rect1
#img[:img_rect2.shape[0], img_rect1.shape[1]:] = img_rect2

plt.savefig("rectified_images_02-05.jpg")
plt.show()
