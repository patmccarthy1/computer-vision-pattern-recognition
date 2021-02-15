import numpy as np
import cv2
import glob
import yaml

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((4*4,3), np.float32)
objp[:,:2] = np.mgrid[0:4,0:4].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('*.jpg')

d = 1

with open('results/images_used.txt', 'w') as outfile:
    outfile.write('Images Used in Calibration: \n')
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (4,4),None)
        #print(ret)

        # If found, add object points, image points (after refining them)
        if ret == True:
            outfile.write(str(d) + ': ' + fname + '\n')
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (4,4), corners2, ret)
        
            cv2.imshow('img',img)
            cv2.waitKey(1000)

            filename = "results/calibration_%d.jpg"%d
            cv2.imwrite(filename, img)
            d+=1

print("Number of images used for calibration: ", d-1)
cv2.destroyAllWindows()

# calibration
ret, mtx, dist, rv, tv = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

rvecs = np.array(rv) 
rvecs_reshaped = rvecs.reshape(rvecs.shape[0], -1) 
tvecs = np.array(tv) 
tvecs_reshaped = tvecs.reshape(tvecs.shape[0], -1) 

# Displayig required output 
print(" Camera matrix:") 
print(mtx) 

print("\n Distortion coefficient:") 
print(dist)

print("\n Rotation Vectors:") 
print(rvecs_reshaped) 

print("\n Translation Vectors:") 
print(tvecs_reshaped) 

# How accurate are the parameters?
tot_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    tot_error += error
mean_error = tot_error/len(objpoints)
print ("Mean error: ", mean_error)

# Save camara parameters to text file
with open('results/camera_parameters.txt', 'w') as outfile:
    outfile.write('# Camera matrix (Intrinsic Parameters: focal length and optical centers): \n')
    np.savetxt(outfile, mtx, fmt='%-7.4f')

    outfile.write('\n# Distortion coefficients: \n')
    np.savetxt(outfile, dist, fmt='%-7.4f')

    outfile.write('\n# Rotation Vectors (extrinsic parameters): \n')
    np.savetxt(outfile, rvecs_reshaped, fmt='%-7.4f')

    outfile.write('\n# Translation Vectors (extrinsic parameters): \n')
    np.savetxt(outfile, tvecs_reshaped, fmt='%-7.4f')

    outfile.write('\n# Mean error (how accurate are the parameters?): ' + str(mean_error))

# In case of distortion:
'''
img = cv2.imread('*.jpg')
h,  w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('undist_image.jpg',dst)
'''
