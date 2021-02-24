#  Computer Vision Coursework

Computer vision algorithms written for the module 'Computer Vision & Pattern Recognition' in Imperial EEE.

## Task 1: Collect Data
1. Collect a sequence of 5-10 pictures (we call it FD) with and without the object in the grid. Change the camera position between pictures.
2. Collect a similar sequence (we call it HG) by changing the zoom (e.g. factor 1.5) and slightly rotating the camera (e.g. 10-20 degree) but keeping exactly the same location
of the camera.

## Task 2: Keypoint orrespondences between images
1. Compare quality/quantity of correspondences found by two methods
    - Manual (clicking on corresponding points)
    - Automatic (detecting keypoint and matching descriptors)

## Task 3: Camera calibration
1. Find and report camera parameters.
2. Can you estimate or illustrate distortions of your camera?

## Task 4: Transformation estimation
1. Estimate a homography matrix between a pair of images from HG.
    - Show the keypoints and their correspondences projected from the other image.
2. Estimate fundamental matrix between a pair of images from FD.
    - Show the keypoints and their corresponding epipolar lines in the other image.
    - Show epipoles, vanishing points and horizon in your images.
   
## Task 5: 3D geometry
1. Show a stereo rectified pair of your images with epipolar lines.
2. Calculate and display depth map of your object estimated from different views.

# Authors
- Maria Arranz Fombellida, ma8816, CID:01250685
- Patrick McCarthy, pm4617, CID:01353165
