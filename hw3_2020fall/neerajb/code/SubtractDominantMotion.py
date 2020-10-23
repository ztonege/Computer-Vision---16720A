import numpy as np
from LucasKanadeAffine import LucasKanadeAffine
from LucasKanadeAffine import warp
from InverseCompositionAffine import InverseCompositionAffine
import scipy.ndimage
import cv2

def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """

    # put your implementation here
    mask = np.zeros(image1.shape, dtype=bool)

    # LK Affine
    #M = LucasKanadeAffine(image1, image2, threshold, num_iters)

    # LK Inverse Compositon Affine
    M = InverseCompositionAffine(image1, image2, threshold, num_iters)

    # warp image 
    warped = warp(image2, M)

    # warp mask and use to catch pixels that fell outside pixel frame
    warped_mask = np.full((image1.shape[0], image1.shape[1]), 255, dtype=np.uint8)
    warped_mask = warp(warped_mask, M)

 
    # params for ants -- comment out for aerial 
    warped = scipy.ndimage.morphology.binary_dilation(warped)
    warped = scipy.ndimage.morphology.binary_erosion(warped)
    
    # look at difference 
    diff = image1 - warped

    # iterate through each pixel of the difference between
    # image1 and the warped image. If the difference is above 
    # the threshold, this indicates that pixel was in motion
    for row in range(image1.shape[0]): #y
        for col in range(image1.shape[1]): #x
            # dont deal with pixels outside frame
            if warped_mask[row, col] == 255:
                if abs(diff[row, col]) > tolerance:
                    mask[row,col] = True
                else:
                    mask[row,col] = False

    #params for ants -- comment out for aerial 
    mask = scipy.ndimage.morphology.binary_erosion(mask)
    mask = scipy.ndimage.morphology.binary_dilation(mask, iterations=2)

    return mask