import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters 
import skimage.morphology 
import skimage.segmentation

import matplotlib.pyplot as plt

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []

    #image = skimage.color.rgb2gray(image)

    thresh = skimage.filters.threshold_otsu(image)
    
    bw = skimage.morphology.closing(image < thresh, skimage.morphology.square(3))

    cleared = skimage.segmentation.clear_border(bw)

    label_image = skimage.measure.label(cleared)

    image_label_overlay = skimage.color.label2rgb(label_image, image=image, bg_label=0)

    for region in skimage.measure.regionprops(label_image):
        # take regions with large enough areas
        if region.area >= 100:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            bboxes.append(region.bbox)

    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions

    bw = 1 - bw 
    return bboxes, bw