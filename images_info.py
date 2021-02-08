import cv2
 
# read image
img = cv2.imread('HG_06.jpg', cv2.IMREAD_UNCHANGED)
 
# get dimensions of image
dimensions = img.shape
 
# height, width, number of channels in image
height = img.shape[0]
width = img.shape[1]
channels = img.shape[2]
 
print('Image Dimension    : ',dimensions)
print('Image Height       : ',height)
print('Image Width        : ',width)
print('Number of Channels : ',channels)


# RESULTS
'''
Image Dimension    :  (284, 378, 3)
Image Height       :  284
Image Width        :  378
Number of Channels :  3
'''