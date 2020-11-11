import numpy as np
import submission
import helper
import cv2
from submission import epipolarCorrespondence
from helper import displayEpipolarF
from helper import epipolarMatchGUI
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D

np.set_printoptions(suppress=True)

# load in correspondences, images and intrinsics 
coords = np.load('../data/templeCoords.npz')
cooresp = np.load('../data/some_corresp.npz')
intrinsics = np.load('../data/intrinsics.npz')
im1 = mpimg.imread('../data/im1.png')
im2 = mpimg.imread('../data/im2.png')

x1 = (coords['x1'])
y1 = (coords['y1'])

pts1 = cooresp['pts1']
pts2 = cooresp['pts2']

K1 = intrinsics['K1']
K2 = intrinsics['K2']

width  = im1.shape[0]
height = im1.shape[1]

# calc normalization value
M = max(width, height)

# fundamental matrix
F = submission.eightpoint(pts1, pts2, M)

pts_original = np.hstack((x1, y1))

pts_secondary = []

for i in range(len(pts_original)):
    res = epipolarCorrespondence(im1, im2, F, pts_original[i][0], pts_original[i][1])
    pts_secondary.append(res)

pts_secondary = np.asarray(pts_secondary)

# essential matrix
E = submission.essentialMatrix(F, K1, K2)

# camera matrix (3,3,4)
M = helper.camera2(E)

# actual camera matrices 
M1 = np.hstack( ( np.eye(3), np.zeros((3, 1) ) ) )
C1 = np.dot(K1, M1)


for i in range(len(M)):
    C2 = np.dot(K2, M[:,:,i])
    temp_P = submission.triangulate(C1, pts_original, C2, pts_secondary)
    if np.all(temp_P[:,2] > 0):
        M2_final = M[:,:,i]
        C2_final = C2
        P_final = temp_P

np.savez("q3_3.npz",M2_final, C2_final, P_final)
