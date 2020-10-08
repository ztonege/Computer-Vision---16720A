import numpy as np
import cv2
from matchPics import matchPics
from opts import get_opts
import scipy.ndimage
from helper import plotMatches
import matplotlib.pyplot as plt

# get options for BRIEF and FAST
opts = get_opts()
ratio = opts.ratio  #'ratio for BRIEF feature descriptor'
sigma = opts.sigma  #'threshold for corner detection using FAST feature detector'

#Q2.1.6
#Read the image and convert to grayscale, if necessary
cv_cover = cv2.imread('../data/cv_cover.jpg')
img_gray = cv_cover
if img_gray.ndim < 3: img = np.stack((img,)*3, axis=-1)

angle = 0
angle_arr = []
count_arr = []
for i in range(36):
	#Rotate Image
	angle -= 10
	angle_sing = np.array([angle])
	img_gray_rot = scipy.ndimage.rotate(img_gray, angle)

	#Compute features, descriptors and Match features
	matches, locs1, locs2 = matchPics(img_gray, img_gray_rot, opts)
	#Update histogram
	angle_arr.append(angle)
	count_arr.append(len(matches))
	#plotMatches(img_gray, img_gray_rot, matches, locs1, locs2)
	
angle_arr = np.asarray(angle_arr)
count_arr = np.asarray(count_arr)

#Display histogram
plt.bar(angle_arr, count_arr, align='center', alpha=0.5)
plt.xticks(angle_arr)
plt.xlabel('Angles')
plt.ylabel('Matches')
plt.title('Angle vs. Matches')
plt.show()

