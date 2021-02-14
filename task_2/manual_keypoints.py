import matplotlib.pyplot as plt
import numpy as np
import mplcursors
import cv2
import csv

# load images
im1 = cv2.imread('../images/HG_03.jpg')
im2 = cv2.imread('../images/HG_06.jpg')

# plot images with cursor so keypoints can be selected (manually create CSV ile of keypoints)
fig, axes = plt.subplots(ncols=2)
axes[0].imshow(im2, interpolation="nearest", origin="upper")
axes[1].imshow(im1, interpolation="nearest", origin="upper")
mplcursors.cursor()
fig.suptitle("Click anywhere on the image")
plt.show()

# read csv of keypoints

with open('manual_keypoints.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile, delimiter='\t'))
A_labels = data[0]
A_x = [int(i) for i in data[1]]
A_y = [int(i) for i in data[2]]
B_labels = data[3]
B_x = [int(i) for i in data[4]]
B_y = [int(i) for i in data[5]]

# plot first image with keypoints lebelled
fig1, axes1 = plt.subplots(ncols=1)
axes1.imshow(im2, interpolation="nearest", origin="upper")
axes1.scatter(x=A_x, y=A_y, c='lime',s=1)
axes1.get_xaxis().set_visible(False)
axes1.get_yaxis().set_visible(False)
fig1.savefig('manual_single.eps', bbox_inches='tight', pad_inches=0)

# plot both images with keypoints labelled
fig2, axes2 = plt.subplots(ncols=2)
axes2[0].imshow(im2, interpolation="nearest", origin="upper")
axes2[0].scatter(x=A_x, y=A_y, c='lime',s=1)
for i, txt in enumerate(A_labels):
    axes2[0].annotate(txt, (A_x[i], A_y[i]),fontsize=3,color='lime')
axes2[0].get_xaxis().set_visible(False)
axes2[0].get_yaxis().set_visible(False)
axes2[1].imshow(im1, interpolation="nearest", origin="upper")
axes2[1].scatter(x=B_x, y=B_y, c='lime',s=1)
for i, txt in enumerate(B_labels):
    axes2[1].annotate(txt, (B_x[i], B_y[i]),fontsize=3,color='lime')
axes2[1].get_xaxis().set_visible(False)
axes2[1].get_yaxis().set_visible(False)
fig2.savefig('manual_both.eps', bbox_inches='tight', pad_inches=0)

# plot images with corresponding keypoints connected
# match_img = cv2.drawMatches(im1, [A_x,A_y], im2, [B_x,B_y], None)
# cv2.imshow('Matches', match_img)
# cv2.waitKey()