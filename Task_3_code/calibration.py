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
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (4,4),None)
    #print(ret)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (4,4), corners2, ret)
        
        #cv2.imshow('img',img)
        #cv2.waitKey(1000)

        filename = "Calibration/calibration_%d.jpg"%d
        #cv2.imwrite(filename, img)
        d+=1

print("Number of images used for calibration: ", d)
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

# Save camara parameters to text file
with open('Calibration/camera_parameters.txt', 'w') as outfile:
    outfile.write('# Camera matrix (includes focal length and optical centers): \n')
    np.savetxt(outfile, mtx, fmt='%-7.4f')
    
    outfile.write('\n# Distortion coefficients (intrinsic parameters): \n')
    np.savetxt(outfile, dist, fmt='%-7.4f')

    outfile.write('\n# Rotation Vectors (extrinsic parameters): \n')
    np.savetxt(outfile, rvecs_reshaped, fmt='%-7.4f')

    outfile.write('\n# Translation Vectors (extrinsic parameters): \n')
    np.savetxt(outfile, tvecs_reshaped, fmt='%-7.4f')
