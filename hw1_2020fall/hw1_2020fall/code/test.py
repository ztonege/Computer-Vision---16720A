import os, multiprocessing
from os.path import join, isfile

import numpy as np
from PIL import Image
import scipy.ndimage
import skimage.color
import random
import visual_words

from functools import partial
from sklearn import cluster 
from multiprocessing import Pool

a = np.array([1,2,3,4])
b = np.array([1,2,3,4])
hist = [a,b]
c = np.sum(hist, axis=0)


for l in range(0,2+1):
    print("new l")
    height = 310
    width = 444

    M = height//pow(2,l)
    N = width//pow(2,l) 

    #for x in range(0,height,M):
    for y in range(0,width+1,N):
        print(y)