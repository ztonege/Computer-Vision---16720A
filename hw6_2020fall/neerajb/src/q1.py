# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# Dec 2020
# ##################################################################### #

# Imports
import numpy as np
from skimage.io import imread
from skimage.color import rgb2xyz
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from utils import integrateFrankot

def renderNDotLSphere(center, rad, light, pxSize, res):

    """
    Question 1 (b)

    Render a sphere with a given center and radius. The camera is 
    orthographic and looks towards the sphere in the negative z
    direction. The camera's sensor axes are centered on and aligned
    with the x- and y-axes.

    Parameters
    ----------
    center : numpy.ndarray
        The center of the sphere in an array of size (3,)

    rad : float
        The radius of the sphere

    light : numpy.ndarray
        The direction of incoming light

    pxSize : float
        Pixel size

    res : numpy.ndarray
        The resolution of the camera frame

    Returns
    -------
    image : numpy.ndarray
        The rendered image of the sphere
    """

    x_size = res[0] * pxSize
    y_size = res[1] * pxSize

    off_setx = x_size/2
    off_sety = y_size/2

    x_cen, y_cen, z_cen = center

    x = np.linspace(0, x_size, res[0])
    y = np.linspace(0, y_size, res[1])

    xv, yv = np.meshgrid(x, y,  indexing='ij')

    xv =  xv  - off_setx 
    yv = off_sety - yv # because numpy does y backwards...ugh

    is_sphere = xv**2 + yv**2 <= rad**2
    is_sphere = is_sphere.astype(int)

    z = np.sqrt(rad ** 2 - (xv - x_cen) ** 2 - (yv - y_cen) ** 2 ) + z_cen 
    z = np.nan_to_num(z)
    # surface normals of a sphere
    x_n = xv - x_cen 
    y_n = yv - y_cen
    z_n = z  - z_cen

    n_arr = np.stack((x_n, y_n, z_n), axis = -1)

    I = []
    for i in range(res[0]):
        for j in range(res[1]):
            I.append(np.maximum(np.dot(n_arr[i][j], light), 0))
    I = np.array(I).reshape(res[0], res[1])
    I = I * is_sphere 

    plt.imshow(I.T, cmap='Greys_r')
    plt.show()

    image = I   
    return image


def loadData(path = "../data/"):

    """
    Question 1 (c)

    Load data from the path given. The images are stored as input_n.tif
    for n = {1...7}. The source lighting directions are stored in
    sources.mat.

    Paramters
    ---------
    path: str
        Path of the data directory

    Returns
    -------
    I : numpy.ndarray
        The 7 x P matrix of vectorized images

    L : numpy.ndarray
        The 3 x 7 matrix of lighting directions

    s: tuple
        Image shape

    """

    n = 8
    data_arr = []
    for i in range(1,8):
        data = imread(path + "input_" + str(i) + ".tif").astype(np.uint16)
        data_xyz = rgb2xyz(data)
        data_arr.append(data_xyz)
    data_arr = np.array(data_arr)
    
    I = []
    for arr in data_arr:
        I.append(arr[:,:,1].flatten())
    I = np.array(I)
   
    L = np.load(path + "sources.npy").T#.astype(np.uint16)

    s = data_arr[0,:,:,0].shape
 
    return I, L, s


def estimatePseudonormalsCalibrated(I, L):

    """
    Question 1 (e)

    In calibrated photometric stereo, estimate pseudonormals from the
    light direction and image matrices

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P array of vectorized images

    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals
    """
    B = np.linalg.inv(L @ L.T) @ L @ I
    
    return B


def estimateAlbedosNormals(B):

    '''
    Question 1 (e)

    From the estimated pseudonormals, estimate the albedos and normals

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of estimated pseudonormals

    Returns
    -------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals
    '''
 
    albedos = []
    for i in range(B.shape[1]):
        albedos.append(np.linalg.norm(B[:,i]))
    albedos = np.array(albedos)

    
    normals = B/albedos
    
    return albedos, normals


def displayAlbedosNormals(albedos, normals, s):

    """
    Question 1 (f)

    From the estimated pseudonormals, display the albedo and normal maps

    Please make sure to use the `gray` colormap for the albedo image
    and the `rainbow` colormap for the normals.

    Parameters
    ----------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    -------
    albedoIm : numpy.ndarray
        Albedo image of shape s

    normalIm : numpy.ndarray
        Normals reshaped as an s x 3 image

    """
    albedoIm = albedos.reshape(s[0], s[1])
    normals = normals.T.reshape(s[0], s[1], 3)
    # normalize to avoid clipping  
    normalIm = (normals - np.min(normals)) / (np.max(normals) - np.min(normals))

    plt.imshow(albedoIm, cmap = 'gray')
    plt.show()

    plt.imshow(normalIm, cmap = 'rainbow')
    plt.show()

    return albedoIm, normalIm


def estimateShape(normals, s):

    """
    Question 1 (i)

    Integrate the estimated normals to get an estimate of the depth map
    of the surface.

    Parameters
    ----------
    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    ----------
    surface: numpy.ndarray
        The image, of size s, of estimated depths at each point

    """
    toFrankot_x = []
    toFrankot_y = []
    for i in range(normals.shape[1]):
        n = normals[:,i]
        zx = -n[0]/n[2]
        zy = -n[1]/n[2]
        toFrankot_x.append(zx)
        toFrankot_y.append(zy)
    
    toFrankot_x = np.array(toFrankot_x)
    toFrankot_y = np.array(toFrankot_y)
    
    toFrankot_x = toFrankot_x.reshape(s)
    toFrankot_y = toFrankot_y.reshape(s)

    surface = integrateFrankot(toFrankot_x, toFrankot_y)

    return surface


def plotSurface(surface):

    """
    Question 1 (i) 

    Plot the depth map as a surface

    Parameters
    ----------
    surface : numpy.ndarray
        The depth map to be plotted

    Returns
    -------
        None

    """
    s = surface.shape

    X = np.arange(0, s[0], 1)
    Y = np.arange(0, s[1], 1)

    X, Y = np.meshgrid(X, Y, indexing='ij')

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # negative needed to match the color scheme of the writeup
    surf = ax.plot_surface(X, Y, surface, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    plt.show()
    pass


if __name__ == '__main__':


    center = np.array([0, 0, 0])
    rad = 0.75
    light1 = np.array([1, 1, 1]/np.sqrt(3))
    light2 = np.array([1, -1, 1]/np.sqrt(3))
    light3 = np.array([-1, -1, 1]/np.sqrt(3))

    pxSize = 7e-4
    res = np.array(([3840, 2160]))

    renderNDotLSphere(center, rad, light1, pxSize, res)
    renderNDotLSphere(center, rad, light2, pxSize, res)
    renderNDotLSphere(center, rad, light3, pxSize, res)

    I,L,s = loadData()

  
    # print(I.shape)
    # print(L.shape)
    # print(s.shape)
    
    # u is eigenvectors of ATA each column is a basis vector
    # s singular values, how much of basis vector is in transformation, basis vectors in u and v 
    # v is eigenvectors of AAT

    # u, v, vh = np.linalg.svd(I, full_matrices=False)
 
    # print(u.shape)
    # print(v.shape)
    # print(vh.shape)

    # print(np.linalg.matrix_rank(vh))
    # print(v)
   
    B = estimatePseudonormalsCalibrated(I,L)
    albedos, normals = estimateAlbedosNormals(B)
    displayAlbedosNormals(albedos, normals, s)
    surface = estimateShape(normals, s)
    plotSurface(-surface)









