import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts

#Import necessary functions
from matchPics import matchPics
from planarH import computeH_ransac
from planarH import compositeH

#Write script for Q2.2.4
opts = get_opts()

# import images
cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')
hp_cover = cv2.imread('../data/hp_cover.jpg')

matches, locs1, locs2 = matchPics(cv_desk, cv_cover, opts)

new_locs1 = []
new_locs2 = []
for match in matches:
    new_locs1.append(locs1[match[0]])
    new_locs2.append(locs2[match[1]])

# flip x and y columns
new_locs1 = np.asarray(new_locs1)
new_locs1[:,[0, 1]] = new_locs1[:,[1, 0]]
new_locs2 = np.asarray(new_locs2)
new_locs2[:,[0, 1]] = new_locs2[:,[1, 0]]

# compute best H
bestH2to1, inliers = computeH_ransac(new_locs1, new_locs2, opts)
# resize hp cover to same shape as cv cover
hp_cover = cv2.resize(hp_cover, (cv_cover.shape[1],cv_cover.shape[0]))

# display composite image
composite_img  = compositeH(bestH2to1, hp_cover, cv_desk)
cv2.imshow('img', composite_img)
cv2.waitKey(0) 
