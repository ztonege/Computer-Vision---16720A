import numpy as np
from scipy.interpolate import RectBivariateSpline
import scipy 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

def LucasKanadeAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [3x3 numpy array] put your implementation here
    """
    # put your implementation here
    M = np.eye(3).astype(float) 

    # evaluate Jacobian at every pixel
    # derivation from lecture #10
    # pre computed for time efficiency 
    # x y 1 0 0 0 
    # 0 0 0 x y 1 
    Jacobian = np.zeros( (It.shape[0], It.shape[1], 2, 6) ) 
    for row in range(It.shape[0]): #y
        for col in range(It.shape[1]): #x
                pixel_Jacobian = np.array( [ [col, row, 1, 0, 0, 0], 
                                             [0, 0, 0, col, row, 1] ] )
                Jacobian[row, col] = pixel_Jacobian
        
    for i in range(num_iters):

        # warp I with W(x;p)
        affineImage = warp(It1, M)

        # create mask
        # mask must be casted to uint8 or cv2.bitwise_and will not work
        mask = np.full((It.shape[0], It.shape[1]), 255, dtype=np.uint8)
        warped_mask = warp(mask, M)


        # create template
        template = cv2.bitwise_and(It, It, mask=warped_mask)

        # get gradient at each pixel
        sobelx = cv2.Sobel(It1,cv2.CV_64F,1,0,ksize=3)
        sobely = cv2.Sobel(It1,cv2.CV_64F,0,1,ksize=3)
        x_derivative = warp(sobelx, M)
        y_derivative = warp(sobely, M)

        #compute hessian using gradient and Jacobian
        H = np.zeros((6,6))
        gradient = np.zeros( (It.shape[0], It.shape[1], 1, 2) ) 
        for row in range(It.shape[0]): #y
            for col in range(It.shape[1]): #x
                if warped_mask[row, col] == 255:
                    gradient[row, col] = np.array( [x_derivative[row, col], y_derivative[row, col] ] )
                    temp_gradient = gradient[row, col]
                    temp_Jacobian = Jacobian[row, col]
                    ATA = np.dot(np.dot(temp_gradient, temp_Jacobian).reshape(6,1), np.dot(temp_gradient, temp_Jacobian))
                    H += ATA

       # get inverse of Hessian 
        H_inv = np.linalg.inv(H)

        # compute delta_p
        delta_p = np.zeros((6,1))
        for row in range(It.shape[0]): #y
            for col in range(It.shape[1]): #x
                if warped_mask[row, col] == 255:
                    temp_gradient = gradient[row, col]
                    temp_Jacobian = Jacobian[row, col]
                    tmp_var = np.dot(temp_gradient, temp_Jacobian).reshape(6,1)
                    delta_p += np.dot(H_inv, tmp_var) * (template[row, col] - affineImage[row, col]) 

        # update M
        M[0,:] = M[0,:] - delta_p[0:3].reshape(3,)
        M[1,:] = M[1,:] - delta_p[3:].reshape(3,)

        # check if delta_p is below the threshold
        if np.linalg.norm(delta_p) < threshold:
            print("Took " + str(i) + " attempts")
            return M
        elif i == num_iters:  
            print("Unable to converge")
        
    return M

def warp(It, M):
    rot_matrix = M[0:2,:]
    warp = warp = cv2.warpAffine(It, rot_matrix, (It.shape[1], It.shape[0]))
    #warp = cv2.warpAffine(It, rot_matrix, (It.shape[1], It.shape[0]))

    ###########################################
    # best attempt to get affine warp to work
    ###########################################
    # warp_x = []
    # warp_y = []
    
    # for row in range(It.shape[0]): #y
    #     for col in range(It.shape[1]): #x
    #         if row > It.shape[0] or col > It.shape[1]:
    #             continue
    #     dotM = np.dot(rot_matrix, np.array([row, col, 1]))


    #     warp_x.append(dotM[0])
    #     warp_y.append(dotM[1])

    # warp_x = np.array(warp_x)
    # warp_y = np.array(warp_y)

    # It_spline=RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It)
    # warp = It_spline(warp_x, warp_y)
    # warp = np.asarray(warp, dtype = np.uint8)


    
    return warp
