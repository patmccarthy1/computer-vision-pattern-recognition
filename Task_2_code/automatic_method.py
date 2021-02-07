import numpy as np
import cv2
from matplotlib import pyplot as plt

# windows to display image
cv2.namedWindow("Image")
# read image
image = cv2.imread('FD_01.png')
# show image
cv2.imshow("Image", image)
# exit at closing of window
cv2.waitKey(0)
cv2.destroyAllWindows()