import submission
import numpy as np
import helper
import matplotlib.image as mpimg

seq = np.load('../data/some_corresp_noisy.npz')
pts1 = seq['pts1']
pts2 = seq['pts2']

im1 = mpimg.imread('../data/im1.png')
im2 = mpimg.imread('../data/im2.png')

width  = im1.shape[0]
height = im1.shape[1]
M = max(width, height)

F,inliers = submission.ransacF(pts1, pts2, M)
print("F", F)
print("Inliers", inliers)
helper.epipolarMatchGUI(im1, im2, F)