# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# Dec 2020
# ##################################################################### #

import numpy as np
from q1 import loadData, estimateAlbedosNormals, displayAlbedosNormals
from q1 import estimateShape, plotSurface 
from utils import enforceIntegrability
from matplotlib import pyplot as plt
from q1 import * 


def estimatePseudonormalsUncalibrated(I):

    """
    Question 2 (b)

    Estimate pseudonormals without the help of light source directions. 

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P matrix of loaded images

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals

    """

    B = None
    L = None

    u, s, vh = np.linalg.svd(I, full_matrices=False)
    # set all but top three singular values to zero
    s = s[:3]
    u = u[:, :3]
    
    # without factorization
    # L = u * s
    # L = L.T

    # with factorization 
    L = u * np.sqrt(s)
    L = L.T
    

    #without factorization
    B = vh[:3,:]
    #with factorization
    B = np.sqrt(s).reshape(3, 1) * B

    return B, L

if __name__ == "__main__":

    # Put your main code here

    I,L0,s = loadData()
    B, L = estimatePseudonormalsUncalibrated(I)

    # print(L0)
    # print(L)
    
    albedos, normals = estimateAlbedosNormals(B)
    displayAlbedosNormals(albedos, normals, s)

    # #without integrability 
    surface = estimateShape(normals, s)
    plotSurface(surface)

    # #enforcing integrability 
    surface_enforced = estimateShape(enforceIntegrability(B, s), s)
    plotSurface(-surface_enforced)

    # bas - relief 
    B_new = enforceIntegrability(B, s)
    G = np.eye(3)
    G[2,:] = [0,0,.005]
    G_inv = np.linalg.inv(G)
    Bnew  = G_inv.T @ B_new
    albedos, normals = estimateAlbedosNormals(Bnew)
    surface = estimateShape(normals, s)
    plotSurface(-surface)



