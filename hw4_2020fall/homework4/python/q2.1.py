import numpy as np
import cv2
import matplotlib.pyplot as pyplot
import matplotlib.image as mpimg
from submission import eightpoint
from helper import displayEpipolarF

# load in correspondences and images 
#cooresp = np.load('../data/some_corresp.npz')
cooresp = np.load('../data/some_corresp_noisy.npz')
im1 = mpimg.imread('../data/im1.png')
im2 = mpimg.imread('../data/im2.png')
pts1 = cooresp['pts1']
pts2 = cooresp['pts2']

width  = im1.shape[0]
height = im1.shape[1]

# calc normalization value
M = max(width, height)

F = eightpoint(pts1, pts2, M)

np.savez("q2_1.npz", F, M)

displayEpipolarF(im1, im2, F)