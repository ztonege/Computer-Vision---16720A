import numpy as np
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2


def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2), flag = 0):
    
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """ 
    # for template correction, we want to make sure we keep the original rect, 
    # therefore warp with a p of [0,0]. Otherwise take the normal p and warp
    p = p0
    p_for_flag_0 = np.array([0,0]).astype(float)

    # changed to deal with using the original template for calculating Pn of frame 0 for template correction
    if flag == 1:
        template = warp(It, rect, p_for_flag_0)
    else:
        template = warp(It, rect, p)
    
    for i in range(num_iters):

        # warp I with W(x;p)
        warpedImage = warp(It1, rect, p)

        # get gradient at each pixel
        sobelx = cv2.Sobel(It1,cv2.CV_64F,1,0,ksize=3)
        sobely = cv2.Sobel(It1,cv2.CV_64F,0,1,ksize=3)
        x_derivative = warp(sobelx, rect, p)
        y_derivative = warp(sobely, rect, p)

        # computer gradient
        gradient = np.transpose(np.vstack([x_derivative.flatten(), y_derivative.flatten()]))
        
        # evaluate Jacobian 
        Jacobian = np.identity(2)

        # compute Hessian 
        H = np.dot(np.transpose(np.dot(gradient, Jacobian)), np.dot(gradient, Jacobian))

        # compute delta p 
        delta_p = np.dot(np.linalg.inv(H), np.transpose(np.dot(gradient, Jacobian)))
        flattened_template = template.flatten()
        flattened_warpedImage = warpedImage.flatten()
        delta_p = np.dot(delta_p, flattened_template - flattened_warpedImage)

        # update p <-- p + delta p
        p = p + delta_p

        # check if delta_p is below threshold
        if np.sqrt(delta_p[0] ** 2 + delta_p[1] ** 2) < threshold:
            print("Took " + str(i) + " attempts")
            return p
        elif i == num_iters:  
            print("Unable to converge")

    return p

def warp(It, rect, p):
    # create new top left and bottom right of rectangle
    top_left = np.array([rect[0] + p[0], rect[1] + p[1]])
    bottom_right = np.array([rect[2] + p[0], rect[3] + p[1]])

    # compute width  and heigh
    width = int(abs(rect[2] - rect[0]))
    height = int(abs(rect[3] - rect[1]))

    # create new coordinates for spline using shifted rect
    x_positions = np.linspace(top_left[0], bottom_right[0], width)
    y_positions = np.linspace(top_left[1], bottom_right[1], height)

    # create spline to represent partial coordinates 
    It_spline=RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It)
    warp = It_spline(y_positions, x_positions)

    return warp

#################################################
# failed attempt to get derivatives using spline, 
# switched to sobel in interest of time
#################################################

# def getDerivative(It, rect, p):

#     It_spline=RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It)
#     width = abs(rect[2] - rect[0])
#     height = abs(rect[3] - rect[1])

#     xcoor_indices = np.arange(width)
#     ycoor_indices = np.arange(height)
#     x_cord, y_cord = np.meshgrid(xcoor_indices, ycoor_indices)

#     derivative_x = It_spline.ev(y_cord, x_cord, dx = 0, dy = 1)
#     derivative_y = It_spline.ev(y_cord, x_cord, dx = 1, dy = 0)
 

#     return derivative_x, derivative_y