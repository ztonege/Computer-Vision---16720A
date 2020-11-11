"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
import numpy as np
import numpy.linalg as LA
from util import refineF
import cv2
import skimage.color
from scipy import signal
import util

'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
  
    # instantiate U
    U = np.zeros(([pts1.shape[0], 9]), dtype=np.float64)

    # crate normalization matrix with no shifting
    T = np.array(([[1/M, 0,   0], 
                   [0,  1/M,  0], 
                   [0,   0,   1]]))

    # normalize points
    pts1_norm = pts1/M
    pts2_norm = pts2/M

    for row in range(len(U)):
        # create normalized points
        xL, yL, zL = np.dot(T, np.array(([pts2[row][0], pts2[row][1], 1])))
        xR, yR, zR = np.dot(T, np.array(([pts1[row][0], pts1[row][1], 1])))

        # populate U array     
        row2add = np.array(([xL*xR, xL*yR, xL, yL*xR, yL*yR, yL, xR, yR, 1]), dtype=np.float64)
        U[row] = row2add
    
    # ATA and find eigenvector cooresponding to least eigen value
    # *note* I pulled this code from my HW2 planarH.py file
    A = np.dot(np.transpose(U), U)
    w,v = LA.eig(A)
    min_eig_index = np.argmin(w)
    min_eig_vector = v[:,min_eig_index]
    F_temp = min_eig_vector.reshape(3,3).astype(float)
    F_refined = refineF(F_temp, pts1_norm, pts2_norm)
    
    #Denormalization
    T_ = np.transpose(T)
    res = np.dot(T_, F_refined)
    F = np.dot(res, T)

    
    np.savez("q2_1.npz", F, M)
    
    return F


'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):

    
    return np.dot(np.dot(np.transpose(K2), F), K1)


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    
    # convert pts1 and pts2 to HC

    pts1_HC = np.hstack( (pts1, np.ones( (len(pts1), 1) ) ) )
    pts2_HC = np.hstack( (pts2, np.ones( (len(pts2), 1) ) ) )

    c1_0 = C1[:,0].reshape(3,1)
    c1_1 = C1[:,1].reshape(3,1)
    c1_2 = C1[:,2].reshape(3,1)
    c1_3 = C1[:,3].reshape(3,1)

    c2_0 = C2[:,0].reshape(3,1)
    c2_1 = C2[:,1].reshape(3,1)
    c2_2 = C2[:,2].reshape(3,1)
    c2_3 = C2[:,3].reshape(3,1)

    err = 0

    res = np.zeros(([len(pts1_HC), 3]))

    for i in range(len(pts1_HC)):
        
        x1_i = pts1_HC[i]
        x2_i = pts2_HC[i]

        c1_0_ = np.transpose( np.cross(np.transpose(x1_i), np.transpose(c1_0) ) )
        c1_1_ = np.transpose( np.cross(np.transpose(x1_i), np.transpose(c1_1) ) )
        c1_2_ = np.transpose( np.cross(np.transpose(x1_i), np.transpose(c1_2) ) )
        c1_3_ = np.transpose( np.cross(np.transpose(x1_i), np.transpose(c1_3) ) )

        c2_0_ = np.transpose( np.cross(np.transpose(x2_i), np.transpose(c2_0) ) )
        c2_1_ = np.transpose( np.cross(np.transpose(x2_i), np.transpose(c2_1) ) )
        c2_2_ = np.transpose( np.cross(np.transpose(x2_i), np.transpose(c2_2) ) )
        c2_3_ = np.transpose( np.cross(np.transpose(x2_i), np.transpose(c2_3) ) )

        # consolidate and remove redundent row
        A_1 = np.hstack((c1_0_, c1_1_, c1_2_, c1_3_))
        A_1 = A_1[:2, :]
    
        A_2 = np.hstack((c2_0_, c2_1_, c2_2_, c2_3_))
        A_2 = A_2[:2, :]

        A = np.vstack((A_1, A_2))

        # take lowest eigenvector cooresponding to lowest eigen value
        u, v, vh = np.linalg.svd(A, full_matrices=False)
        lowest_row = vh[3,:]
  
        td_point = (lowest_row/lowest_row[3]).reshape(4, 1)
        res[i] = td_point[0:3].reshape(3)

        # denormalize for sake of checking error value 
        x1_guess = np.dot(C1, td_point/td_point[2]).flatten()
        x2_guess = np.dot(C2, td_point/td_point[2]).flatten()

        x1_guess_ = x1_guess/x1_guess[2]
        x2_guess_ = x2_guess/x2_guess[2]

        err += np.linalg.norm(x1_i - x1_guess_) ** 2 + \
               np.linalg.norm(x2_i - x2_guess_) ** 2

    #print(err)
    
    return res


'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    
    box_size = 30
    x_max = im1.shape[1]
    y_max = im1.shape[0]

    # create box for im1
    box_outline_im1 = [y1 - box_size, y1 + box_size, x1 - box_size, x1 + box_size]
    
    im1_grey = skimage.color.rgb2gray(im1)
    im2_grey = skimage.color.rgb2gray(im2)
    
    # collect pixel intensities within box of im1
    # grey using gaussian filter as suggested by:
    # https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
    im1_data = im1_grey[box_outline_im1[0]:box_outline_im1[1], \
                        box_outline_im1[2]:box_outline_im1[3]]

    
    g = signal.gaussian(box_size * 2, std = 15).reshape(box_size * 2, 1)
    blur = np.outer(g,g)
    im1_data_blurred = im1_data * blur

    x2 = int(x1)
    y2 = int(y1)
    v = np.array([x2, y2, 1])
    l = F.dot(v)
    s = np.sqrt(l[0]**2+l[1]**2)
    l = l/s

    # only check within a 15 pixel bound to avoid falling into false positives 
    bounds = 15
    num_steps = 100
    ye = y1 + bounds
    ys = y1 - bounds

    # traverse line
    y_walk = np.linspace(ys, ye, num_steps)
    
    most_similiar = np.inf

    for y_step in y_walk:

        # calc x position based of y using equation of epipolar line
        x_step = -(l[1] * y_step + l[2])/l[0]

        # make sure point doesn't fall outside of image
        if round(x_step) - box_size < 0 or round(x_step) + box_size > x_max or round(y_step) - box_size < 0 or round(y_step) + box_size > y_max :
            res = np.array(([x_step, y_step]))
            continue

        else:
            # create box for im2
            box_outline_im2 = [round(y_step) - box_size, round(y_step) + box_size, \
                            round(x_step) - box_size, round(x_step) + box_size]
            
            # collect pixel intensities within box of im2
            im2_data = im2_grey[box_outline_im2[0]:box_outline_im2[1], \
                                box_outline_im2[2]:box_outline_im2[3]]


            im2_data_blurred = im2_data * blur

            # calc euclidian distance 
            dist = np.linalg.norm(im2_data_blurred-im1_data_blurred)

            # update most viable point 
            if dist < most_similiar:
                most_similiar = dist
                res = np.array(([x_step, y_step]))
    # print("input",x1,y1 ) 
    # print("output", res)      
    return res

'''
Q5.1: Extra Credit RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M, nIters=1000, tol=0.42):
    nIters = 200
    count = 0
    count_ones = 0
    bestF = np.zeros((3, 3))
    inliers = np.zeros((len(pts1)))
    # convert pts to homogenous coordinates 
    pts1_HC = np.hstack( (pts1, np.ones( (len(pts1), 1) ) ) )
    pts2_HC = np.hstack( (pts2, np.ones( (len(pts2), 1) ) ) )
    for i in range(nIters):
        print("Running Iteration: ",i)
        # generate 8 random pts
        rnd = np.random.choice(len(pts1), 8, replace=False)
        # generate F
        F = eightpoint(pts1[rnd], pts2[rnd], M)
        within_tol = np.zeros(len(pts1))

        for j in range(len(pts1)):
            # calculate point 
            line = np.dot(F, pts1_HC[j])
            # calculate distance of point to epipolar line
            dist = np.sqrt(line[0] ** 2 + line[1] ** 2)
            point = pts2_HC[j]
            line = abs(point[0]*line[0] + point[1]*line[1] + point[2]*line[2])
            # if err less than tolerance, include as inlier 
            err = line / dist
            if tol >= err: 
                within_tol[j] = 1
            
        count_ones = np.count_nonzero(within_tol)

        if count_ones > count:
            bestF = F
            inliers = within_tol
            count = count_ones
            
    return bestF, inliers


'''
Q5.2:Extra Credit  Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # Replace pass by your implementation
   
    theta = np.linalg.norm(r)
    w = r/theta 

    kx = w[0][0]
    ky = w[1][0]
    kz = w[2][0]

    
    K = np.array(([[0, -kz, ky],
                   [kz, 0, -kx],
                   [-ky, kx, 0]]))

    I = np.eye(3)
    R = I + np.dot(np.sin(theta),K) + np.dot((1-np.cos(theta)), np.matmul(K,K))

    print(R)

    return R

'''
Q5.2:Extra Credit  Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):

  
    theta = np.arccos((np.trace(R) - 1) /2)

    R32 = R[2][1]
    R23 = R[1][2]

    R13 = R[0][2]
    R31 = R[2][0]

    R21 = R[1][0]
    R12 = R[0][1]

    w = 1/(2*np.sin(theta)) * np.array(([[R32 - R23], 
                                         [R13 - R31],
                                         [R21 - R12]]))
    

    r = theta * w
   
    return r


'''
Q5.3: Extra Credit Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    pass

'''
Q5.3 Extra Credit  Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation
    pass
