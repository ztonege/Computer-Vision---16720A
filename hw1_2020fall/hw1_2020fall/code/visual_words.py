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


def extract_filter_responses(opts, img):
    '''
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''
    
    filter_scales = opts.filter_scales

    # if image is grayscale, convert to 3-channel
    if img.ndim < 3: 
        img = np.stack((img,)*3, axis=-1)

    # if values are not between 0 and 1, normalize 
    maxval = np.amax(img)
    minval = np.amin(img)
    if maxval > 1 or minval < 0:
        img = (img - minval) / (maxval - minval)

    # if img is not of type float, cast to float
    if img.dtype != 'float32': 
        img = img.astype(float)

    # collect dimension of img 
    shape = np.shape(img)
    height = shape[0]
    width = shape[1]
    depth = shape[2]

    img = skimage.color.rgb2lab(img)

    filter_responses = np.ndarray((height,width,0))    

    # apply four filters across each pixel channel 
    for sigma in filter_scales:

        g_filtered = np.ndarray((height,width,0))   
        for i in range(depth):
            channel = img[:,:,i] 
            # apply gaussian filter
            temp = scipy.ndimage.gaussian_filter(channel, sigma)
            temp = temp[...,np.newaxis] # add third dimension for concatenation 
            g_filtered = np.concatenate((g_filtered, temp), axis=2)
     
        gl_filtered = np.ndarray((height,width,0))  
        for i in range(depth):
            channel = img[:,:,i] 
            # apply gaussian laplace filter
            temp = scipy.ndimage.gaussian_laplace(channel, sigma)
            temp = temp[...,np.newaxis] # add third dimension for concatenation 
            gl_filtered = np.concatenate((gl_filtered, temp), axis=2)
        
        gx_filtered = np.ndarray((height,width,0))  
        for i in range(depth):
            channel = img[:,:,i] 
            # apply horizontal gaussian filter 
            temp = scipy.ndimage.gaussian_filter(channel, sigma, (0,1))
            temp = temp[...,np.newaxis] # add third dimension for concatenation 
            gx_filtered = np.concatenate((gx_filtered, temp), axis=2)

        gy_filtered = np.ndarray((height,width,0)) 
        for i in range(depth):
            channel = img[:,:,i] 
            # apply vertical gaussian filter 
            temp = scipy.ndimage.gaussian_filter(channel, sigma, (1,0))
            temp = temp[...,np.newaxis] # add third dimension for concatenation 
            gy_filtered = np.concatenate((gy_filtered, temp), axis=2)

        # concatentate all filters into one array 
        temp_stack = np.concatenate((g_filtered, gl_filtered, gx_filtered, gy_filtered), axis=2)
    
        # add all filters for one channel to final array 
        filter_responses = np.concatenate((filter_responses, temp_stack), axis=2)
            
    return filter_responses

# ADDED INPUT
def compute_dictionary_one_image(args, img_path):
    '''
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    '''
    # isolate the .jpg name
    img_name = img_path.split('/', 1)[-1]
    # load in img and create filter responses
    img_path = join(args.data_dir, img_path)
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32)/255
    filter_responses = visual_words.extract_filter_responses(args, img)
    
    alpha = args.alpha    

    #  collect dimension of filer_responses
    shape = np.shape(filter_responses)
    height = shape[0]
    width = shape[1]
    depth = shape[2]  

    # allocate 'tube' of data for 1 pixel
    tube_array = np.ndarray((0, depth))  
    for i in range(alpha):

        # select random pixels and grab the tube of data
        rand_y = random.randint(0, width) - 1
        rand_x = random.randint(0, height) - 1

        tube = filter_responses[rand_x, rand_y, :]
        tube = tube.reshape(1,depth)

        # append tube to tube array 
        tube_array = np.concatenate((tube_array, tube), axis=0)

    # remove .jpg from image name and save
    img_name = img_name[:-4]
    np.save(join(args.out_dir, 'testdata', img_name), tube_array)
   
    return None

def compute_dictionary(opts, n_worker=1):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel
    
    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    '''

    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()

    f = train_files
    # create partial function to represent compute_dictionary_one_image
    # since pool.map only takes one function as input
    pFunc = partial(compute_dictionary_one_image, opts)
    with Pool(processes=n_worker) as pool:
        pool.map(pFunc, f)

    visual_words_array = []
    
    # collect saved images and pack into one array for K-means
    for img in os.listdir(join(out_dir, 'testdata')):
        path = join(out_dir, 'testdata', img)
        loaded_img = np.load(path)
        for row in loaded_img:
            visual_words_array.append(row)

    # pass visual_words_array to kmeans to save dict of visual words
    visual_words_array = np.asarray(visual_words_array)
    kmeans = cluster.KMeans(n_clusters=K, n_jobs=n_worker).fit(visual_words_array)
    dictionary = kmeans.cluster_centers_
    np.save(join(out_dir, 'dictionary.npy'), dictionary)

    return None

def get_visual_words(opts, img, dictionary):
    '''
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    # extract filter response
    filter_response = visual_words.extract_filter_responses(opts, img)

    # collect dimension of filter_response
    shape = np.shape(filter_response)
    height = shape[0]
    width = shape[1]
    depth = shape[2]  

    # flatten 3D responses into 2D array of size (height*width, depth)
    flattened_filtered_response = filter_response.reshape(height * width, depth)
    
    # calculate distance for each sampled pixel to all words in dict
    distances = scipy.spatial.distance.cdist(flattened_filtered_response, dictionary)
    distances = distances.reshape(height, width, distances.shape[1])
    
    # assign value of pixel in wordmap to index of smallest distance to word
    wordmap = np.ndarray((height, width)) 
    for i in range(height):
        for j in range(width):
            index = np.argmin(distances[i,j])
            wordmap[i][j] = index
  
    return wordmap
   
            
    

    

