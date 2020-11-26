import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# load the weights
# run the crops through your neural network and print them out
import pickle
import string
letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
params = pickle.load(open('q3_weights.pickle','rb'))


for img in os.listdir('../images'):

    #img = '02_letters.jpg'
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    im1 = skimage.color.rgb2gray(im1)

    bboxes, bw = findLetters(im1)
    stored_bboxes = bboxes

    plt.imshow(bw,cmap = "gray")
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()

    bboxes = sorted(bboxes, key = lambda x: x[0])
    bucket_ymin, bucket_ymax = bboxes[0][2], bboxes[0][2]

    sorted_bboxes = []
    bucket = []


    for box in bboxes:
        miny, minx, maxy, maxx = box
        if (miny > bucket_ymax):
            sorted_bboxes.append(bucket)
            bucket = []
            bucket.append(box)
            bucket_ymin, bucket_ymax = box[0], box[2]
            
        else:
            bucket.append(box)
            bucket_ymin, bucket_ymax = box[0], box[2]

    sorted_bboxes.append(bucket)
    final_arr = []
    for bucket in sorted_bboxes:
        final_arr.append(sorted(bucket, key = lambda x: x[1]))

    final_arr_ = []
    for bucket in final_arr:
        for item in bucket:
            final_arr_.append(item)


    assert(len(final_arr_) == len(stored_bboxes))
    sent_guess = []
    # https://stackoverflow.com/questions/10871220/making-a-matrix-square-and-padding-it-with-desired-value-in-numpy
    for box in final_arr_:

        img_data = bw[box[0]:box[2], box[1]:box[3]].astype(np.float64)
        
        (r,c) = img_data.shape

        t = 20
        if (r > c):
            d = r - c
            d_avg = int(d/2)
            padding = ((t,t), (d_avg + t, d_avg + t))
        else:
            d = c - r
            d_avg = int(d/2)
            padding = ((d_avg + t,d_avg + t), (t, t))

        
        img_data = np.pad(img_data, padding , mode = 'constant', constant_values = 1.0)
        # plt.imshow(img_data, cmap = 'gray')
        # plt.show()
        img_resized = skimage.transform.resize(img_data, (32, 32), anti_aliasing=True)

        img_resized = skimage.morphology.erosion(img_resized)

        # plt.imshow(img_resized, cmap = 'gray')
        # plt.show()

        x = img_resized.T.reshape(1, 1024)

        h1 = forward(x,params,'layer1')
        probs = forward(h1,params,'output',softmax)

        guess = letters[np.argmax(probs)]
        #print(guess)
        sent_guess.append(guess)
        
    listToStr = ' '.join([str(elem) for elem in sent_guess]) 
    print(listToStr)




    