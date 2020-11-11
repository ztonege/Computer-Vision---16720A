import numpy as np
import cv2
import submission
import helper
import matplotlib.pyplot as pyplot
import matplotlib.image as mpimg

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

# fundamental matrix
F = submission.eightpoint(pts1, pts2, M)

#displayEpipolarF(im1, im2, F)

# essential matrix
E = submission.essentialMatrix(F, K1, K2)

# camera matrix (3,3,4)
M = helper.camera2(E)

# actual camera matrices 
C1 = np.dot(K1, np.hstack( ( np.eye(3), np.zeros((3, 1) ) ) ) )
#for i in range(len(M)):
    
# test every slice of M, return the coordinate set which places
# all 3D points in front of the camera 
for i in range(len(M)):
    C2 = np.dot(K2, M[:,:,i])
    temp_P = submission.triangulate(C1, pts1, C2, pts2)
    if np.all(temp_P[:,2] < 0):
        M2_final = M[:,:,i]
        P = temp_P
        np.savez("q3_3.npz",M2_final, C2, P)

   