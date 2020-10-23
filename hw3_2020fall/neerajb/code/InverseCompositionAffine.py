import numpy as np
from scipy.interpolate import RectBivariateSpline
import cv2

def InverseCompositionAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [3x3 numpy array]
    """

    # put your implementation here
    M = np.eye(3)

    #precompute gradient of template
    sobelx = cv2.Sobel(It,cv2.CV_64F,1,0,ksize=3)
    sobely = cv2.Sobel(It,cv2.CV_64F,0,1,ksize=3)
    x_derivative = warp(sobelx, M)
    y_derivative = warp(sobely, M)

    # evaluate Jacobian at every pixel
    # x y 1 0 0 0 
    # 0 0 0 x y 1 
    # pre compute jacobian, gradient of template and hessian
    Jacobian_mat = np.zeros( (It.shape[0], It.shape[1], 2, 6) ) 
    gradient = np.zeros( (It.shape[0], It.shape[1], 1, 2) )    
    H = np.zeros((6,6))
    for row in range(It.shape[0]): #y
        for col in range(It.shape[1]): #x
                pixel_Jacobian = np.array( [ [col, row, 1, 0, 0, 0], 
                                             [0, 0, 0, col, row, 1] ] )
                Jacobian_mat[row, col] = pixel_Jacobian
                gradient[row, col] = np.array( [x_derivative[row, col], y_derivative[row, col] ] )
                temp_gradient = gradient[row, col]
                temp_Jacobian = Jacobian_mat[row, col]
                ATA = np.dot(np.dot(temp_gradient, temp_Jacobian).reshape(6,1), np.dot(temp_gradient, temp_Jacobian))
                H += ATA

    H_inv = np.linalg.inv(H)

    for i in range(num_iters):

        # warp I with W(x;p)
        affineImage = warp(It1, M)

            
        # create mask
        # mask must be passed to uint8 or cv2.bitwise_and will not work
        mask = np.full((It.shape[0], It.shape[1]), 255, dtype=np.uint8)
        warped_mask = warp(mask, M)

        template = cv2.bitwise_and(It, It, mask=warped_mask)
        
        delta_p = np.zeros((6,1))
        for row in range(It.shape[0]): #y
            for col in range(It.shape[1]): #x
                if warped_mask[row, col] == 255:
                    temp_gradient = gradient[row, col]
                    temp_Jacobian = Jacobian_mat[row, col]
                    tmp_var = np.dot(temp_gradient, temp_Jacobian).reshape(6,1)
                    #delta_p is now the affine image - the template
                    delta_p += np.dot(H_inv, tmp_var) * (affineImage[row, col] - template[row, col]) 

        # now add delta p vs forward compositional 
        M[0,:] = M[0,:] + delta_p[0:3].reshape(3,)
        M[1,:] = M[1,:] + delta_p[3:].reshape(3,)


        # delta_M = delta_p.reshape(2, 3)
 
        # delta_M[0][0] = delta_M[0][0] + 1
        # delta_M[1][1] = delta_M[1][1] + 1
   
        # #print(inv_delta_p)
        # delta_M = np.vstack((delta_M, np.array([0, 0, 1]).astype(float)))
        # #print(inv_delta_p)
        
        # inv_delta_p = np.linalg.inv(delta_M).astype(float)

        # M[0,:] = np.dot(M[0,:],inv_delta_p[0])
        # M[1,:] = np.dot(M[1,:],inv_delta_p[1])
        
        if np.linalg.norm(delta_p) < threshold:
            print("Took " + str(i) + " attempts")
            return M
        elif i == num_iters:  
            print("Unable to converge")
        
    return M

def warp(It, M):
    rot_matrix = M[0:2,:]
    warp = cv2.warpAffine(It, rot_matrix, (It.shape[1], It.shape[0]))

    return warp


