import numpy as np
import cv2
import matplotlib.pyplot as pyplot
import matplotlib.image as mpimg
from submission import eightpoint
from submission import essentialMatrix
import helper

np.set_printoptions(suppress=True)

# load in correspondences and images 
cooresp = np.load('../data/some_corresp.npz')
im1 = mpimg.imread('../data/im1.png')
im2 = mpimg.imread('../data/im2.png')
pts1 = cooresp['pts1']
pts2 = cooresp['pts2']

# load in camera intrinsics 
intrinsics = np.load('../data/intrinsics.npz')
K1 = intrinsics['K1']
K2 = intrinsics['K2']

width  = im1.shape[0]
height = im1.shape[1]

# calc normalization value
M = max(width, height)

# get fundamental matrix
F = eightpoint(pts1, pts2, M)

#displayEpipolarF(im1, im2, F)

# get essential matrix
E = essentialMatrix(F, K1, K2)

print(F)
print(E)