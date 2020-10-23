import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade
import skimage.color
import copy

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-3, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--template_threshold', type=float, default=10, help='threshold for determining whether to update template')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
template_threshold = args.template_threshold

seq = np.load("../data/carseq.npy")
rect = [59.0, 116.0, 145.0, 151.0]

frame0 = seq[:,:,0]
template_temp  = seq[:,:,0]
p = np.array([0,0]).astype(float)
p_temp = np.array([0,0]).astype(float)
rect_arr = np.zeros(shape=(seq.shape[2],4))

for i in range(1, seq.shape[2]):
    print("Computing Frame: " + str(i))
    temp_rect = copy.copy(rect)
    It = seq[:,:,i-1] #Tx
    It1 = seq[:,:,i] #Ix
    # lucas kanade with Tn+1
    # added flag with will use frame 0 rect for the template, default is zero
    p = LucasKanade(template_temp, It1, rect, threshold, int(num_iters), p0=p_temp)

    # lucas kanade with T1
    pStar = LucasKanade(frame0, It1, rect, threshold, int(num_iters), p0=p, flag = 1)

    if np.linalg.norm(pStar - p) < template_threshold:
        p_temp = pStar
        template_temp = seq[:,:,i]
        print("Using pStar")
    else:
        print("Using p")

    temp_rect = [temp_rect[0] + p[0], temp_rect[1] + p[1], temp_rect[2] + p[0], temp_rect[3] + p[1]]
    rect_arr[i] = temp_rect
    if i == 1 or i == 99 or i == 199 or i == 299 or i == 399:
        fig = plt.figure()
        ax = fig.add_subplot(111) 
        im = plt.imshow(It, cmap='gray')
        rect_frame = patches.Rectangle((temp_rect[0], temp_rect[1]), 
                                temp_rect[2] - temp_rect[0], temp_rect[3] - temp_rect[1], 
                                color ='red',
                                fill = False)
        ax.add_patch(rect_frame)
        plt.xlim([0, seq[:,:,i].shape[1]]) 
        plt.ylim([seq[:,:,i].shape[0], 0]) 
        plt.show() 
np.save('../result/carseqrects-wcrt', rect_arr)
