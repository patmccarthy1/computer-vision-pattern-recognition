import numpy as np
import cv2
import glob
import argparse

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
image_size = None

def load_image_points(left_dir='images/left', right_dir='images/right', width=4, height=4):
    pattern_size = (width, height)  # Chessboard size!
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((height * width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    left_imgpoints = []  # 2d points in image plane.
    right_imgpoints = []  # 2d points in image plane.

    # Get images for left and right directory. Since we use prefix and formats, both image set can be in the same dir.
    left_images = glob.glob(left_dir + '/*.jpg')
    right_images = glob.glob(right_dir + '/*.jpg')

    d = 1 # start a counter from 1

    # Images should be perfect pairs. Otherwise all the calibration will be false.
    # Be sure that first cam and second cam images are correctly prefixed and numbers are ordered as pairs.
    # Sort will fix the globs to make sure.
    left_images.sort()
    right_images.sort()

    pair_images = zip(left_images, right_images)  # Pair the images for single loop handling

    # Iterate through the pairs and find chessboard corners. Add them to arrays
    # If openCV can't find the corners in one image, we discard the pair.
    for left_im, right_im in pair_images:
        # Right Object Points
        right = cv2.imread(right_im)
        gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret_right, corners_right = cv2.findChessboardCorners(gray_right, pattern_size,
                                                             cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FILTER_QUADS)

        # Left Object Points
        left = cv2.imread(left_im)
        gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret_left, corners_left = cv2.findChessboardCorners(gray_left, pattern_size,
                                                           cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FILTER_QUADS)

        if ret_left and ret_right:  # If both image is okay. Otherwise we explain which pair has a problem and continue
            # Object points
            objpoints.append(objp)
            # Right points
            corners2_right = cv2.cornerSubPix(gray_right, corners_right, (5, 5), (-1, -1), criteria)
            right_imgpoints.append(corners2_right)
            # Left points
            corners2_left = cv2.cornerSubPix(gray_left, corners_left, (5, 5), (-1, -1), criteria)
            left_imgpoints.append(corners2_left)
        else:
            print("Chessboard couldn't detected. Image pair: ", left_im, " and ", right_im)
            continue

    image_size = gray_right.shape  # If you have no acceptable pair, you may have an error here.
    return [objpoints, left_imgpoints, right_imgpoints]

def calibrate(dir, width=4, height=4):
    """ Apply camera calibration operation for images in the given directory path. """
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((height*width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob(dir + '/*.jpg')

    d = 1 # start a counter from 1
    # Create a txt file that generates a list of the images that are used for calibration
    with open(dir+'/calibration/images_used.txt', 'w') as outfile:
        outfile.write('Images Used in Calibration: \n') 
        for fname in images: # loop through all the jpg images loaded
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert into gray-scale

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (4,4),None)
            
            # If found, add object points, image points (after refining them)
            if ret == True:
                outfile.write(str(d) + ': ' + fname + '\n') # if true then image name is added onto txt file of used images
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                imgpoints.append(corners2)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (4,4), corners2, ret) # draw the chessboard for a 5x5 grid (4 because starts from 0)

                filename = dir+'/calibration/calibration_%d.jpg'%d # save calibration image with counter as identifier
                cv2.imwrite(filename, img)
                d+=1 # update counter
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return [ret, mtx, dist, rvecs, tvecs]

def save_coefficients(mtx, dist, rv, tv,dir):
    # change vector 3D arrays into 2D arrays to show in txt file
    rvecs = np.array(rv) 
    rvecs_reshaped = rvecs.reshape(rvecs.shape[0], -1) 
    tvecs = np.array(tv) 
    tvecs_reshaped = tvecs.reshape(tvecs.shape[0], -1) 

    # Save camara parameters to text file
    with open(dir+'/calibration/camera_parameters.txt', 'w') as outfile:
        outfile.write('# Camera matrix (Intrinsic Parameters: focal length and optical centers): \n')
        np.savetxt(outfile, mtx, fmt='%-7.4f')
        outfile.write('\n# Distortion coefficients: \n')
        np.savetxt(outfile, dist, fmt='%-7.4f')
        outfile.write('\n# Rotation Vectors (extrinsic parameters): \n')
        np.savetxt(outfile, rvecs_reshaped, fmt='%-7.4f')
        outfile.write('\n# Translation Vectors (extrinsic parameters): \n')
        np.savetxt(outfile, tvecs_reshaped, fmt='%-7.4f')

def load_coefficients(dir):
    """ Loads camera matrix and distortion coefficients. """
    ret, mtx, dist, rvecs, tvecs = calibrate(dir)
    save_coefficients(mtx, dist, rvecs, tvecs,dir)
    print("Calibration is finished. RMS: ", ret)

    camera_matrix = mtx
    dist_matrix = dist
    return [camera_matrix, dist_matrix]

def stereo_calibrate(left_dir='images/left', right_dir='images/right', width=4, height=4):
    """ Stereo calibration and rectification """
    objp, leftp, rightp = load_image_points(left_dir, right_dir)

    K1, D1 = load_coefficients(left_dir)
    K2, D2 = load_coefficients(right_dir)

    flag = 0
    # flag |= cv2.CALIB_FIX_INTRINSIC
    flag |= cv2.CALIB_USE_INTRINSIC_GUESS
    ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(objp, leftp, rightp, K1, D1, K2, D2, image_size)
    print("Stereo calibration rms: ", ret)
    R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(K1, D1, K2, D2, image_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.9)

    return K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q

def depth_map(imgL, imgR):
    # Initialize the stereo block matching object 
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=19)

    # Compute the disparity image
    disparity = stereo.compute(imgL, imgR)

    # Normalize the image for representation
    min = disparity.min()
    max = disparity.max()
    disparity = np.uint8(255 * (disparity - min) / (max - min))

    return disparity

K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q = stereo_calibrate()

# define images to be read in
left_name = '../images/FD_03.jpg'
right_name = '../images/FD_04.jpg'

# read in images 
leftFrame = cv2.imread(left_name, 0) 
rightFrame = cv2.imread(right_name, 0)

height, width = leftFrame.shape

# Undistortion and Rectification part!
leftMapX, leftMapY = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (width, height), cv2.CV_32FC1)
left_rectified = cv2.remap(leftFrame, leftMapX, leftMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
rightMapX, rightMapY = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (width, height), cv2.CV_32FC1)
right_rectified = cv2.remap(rightFrame, rightMapX, rightMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

disparity_image = depth_map(left_rectified, right_rectified)  # Get the disparity map

# Show the images
cv2.imshow('left(R)', leftFrame)
cv2.imshow('right(R)', right_rectified)
cv2.imshow('Disparity', disparity_image)
cv2.waitKey(0)