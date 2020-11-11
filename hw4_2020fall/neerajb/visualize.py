'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''

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

####################
#       q2.1       #
####################

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
  
# test every slice of M, return the coordinate set which places
# all 3D points in front of the camera 

for i in range(len(M)):
    C2 = np.dot(K2, M[:,:,i])
    temp_P = submission.triangulate(C1, pts_original, C2, pts_secondary)
    if np.all(temp_P[:,2] > 0):
        M2_final = M[:,:,i]
        C2_final = C2
        P_final = temp_P

C2 = np.dot(K2, M2_final)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(P_final[:,0], P_final[:,1], P_final[:,2])
plt.show()

np.savez("q4_2.npz", F, M1, M2_final, C1, C2)

