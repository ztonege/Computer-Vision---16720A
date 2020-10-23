import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from LucasKanadeAffine import LucasKanadeAffine
from LucasKanadeAffine import warp
from SubtractDominantMotion import SubtractDominantMotion
import time

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e3, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--tolerance', type=float, default=.1, help='binary threshold of intensity difference when computing the mask')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
tolerance = args.tolerance

seq = np.load('../data/aerialseq.npy')

for i in range(1, seq.shape[2]):
    start_time = time.time()

    print("Computing Frame: " + str(i))

    It = seq[:,:,i-1] # template
    It1 = seq[:,:,i] # image

    mask = SubtractDominantMotion(It, It1, threshold, int(num_iters), tolerance)

    # create a three channe image for It1
    #channeled_img = cv2.merge((It1, It1, It1))
    channeled_img = cv2.merge((It, It, It))

    # for any location in the mask with a value of True,
    # boost the blue channel so we can highlight the movement points
    for row in range(It1.shape[0]): #y
        for col in range(It1.shape[1]): #x
            if mask[row, col] == True:
                channeled_img[row,col,2] = channeled_img[row,col,2] + 100
                channeled_img[row,col,1] = channeled_img[row,col,1] - .2
                channeled_img[row,col,0] = channeled_img[row,col,0] - .2

    current_time = time.time()
    print(current_time - start_time)

    if  i == 30 or i == 60 or i == 90 or i == 120:
        # fig = plt.subplot(1,2,1)
        # im = plt.imshow(It, cmap='gray')
        fig = plt.figure()
        im = plt.imshow(channeled_img, cmap='gray')
        plt.savefig('../result/aerielseqInverse_' + str(i) + '.png')
        #plt.show() 