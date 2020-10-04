import os, math, multiprocessing
from os.path import join
from copy import copy

import numpy as np
from PIL import Image

import visual_words

import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial

def get_feature_from_wordmap(opts, wordmap):
    '''
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    '''

    # return histogram for single image based on word map and normalize
    K = opts.K
    hist, bins = np.histogram(wordmap, bins=K, range=(0,K))
    hist = hist/np.sum(hist)
  
    return hist

def get_feature_from_wordmap_SPM(opts, wordmap):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^L-1)/3)
    '''
        
    K = opts.K
    L = opts.L
    # conversion to int to handle when evaluate_recognition_system 
    # converts L to a numpy array
    L = int(L)

    hist_all = np.ndarray((0))

    # collect dimension of wordmap 
    shape = np.shape(wordmap)
    height = shape[0] 
    width = shape[1] 

    # since not all images are perfect squares when 'tile-ing' the image, 
    # this line supresses warnings about jagged arrays being created
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 

    # according to piazza, L equals the total number of layers of the pyramid, NOT L+1
    
    for l in range(1,L+1):

        hist_layer = np.ndarray((0))

        # determine the dimensions of the 'tiles'
        # +1 to avoid falling on the outermost indices
        M = height//pow(2,l-1) + 1
        N = width//pow(2,l-1) + 1
        
        # break the image into 2^l images
        tiles = np.ndarray((0))
        # list comprehension code snippet from https://stackoverflow.com/questions/5953373/how-to-split-image-into-multiple-pieces-in-python
        tiles = [wordmap[x:x+M,y:y+N] for x in range(0,height,M) for y in range(0,width,N)]
        tiles = np.asarray(tiles)

        # shapes those tiles into a 1D array for each specific layer 
        for tile in tiles:
            hist_layer = np.append(hist_layer, get_feature_from_wordmap(opts, tile))
        
        # weigh each layer by appropriate weight
        # these formulas do NOT follow the writeup but in order to follow the
        # guidelines on Piazza I had to change the eqation to 2**(-L+1) 
        if l == 1:
            weight = pow(2,-L+1)
        else:
            weight = pow(2, (l-L-1))
    
        # normalize each layer so it's sum is equal to it's weight 
        hist_layer = hist_layer/np.sum(hist_layer)
        hist_layer =  hist_layer*weight

        # the normalized value of the concatenated layers should equal 1
        hist_all = np.append(hist_all, hist_layer)
    
        ## uncomment to see individual size of tiles - debugging code
        
        # print("length of layer " + str(l) + " is " + str(hist_layer.shape[0]))
        # print("the size of one square for layer " + str(l) + " is " + str(tiles[0].shape))

    return hist_all

    
def get_image_feature(opts, img_path, dictionary):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K)
    '''

    # helper function to extract_single_image_features, 
    # calls get_feature_from_wordmap_SPM
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32)/255
    wordmap = visual_words.get_visual_words(opts, img, dictionary)
    return get_feature_from_wordmap_SPM(opts, wordmap)

def extract_single_image_features(opts, dictionary, training_files_plus_label):

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    # extract individual name and label 
    (name, label) = training_files_plus_label
    img_path = join(data_dir, name)

    # add word onto front of label name and remove .jpg
    # save file with newName to features folder 
    newName = str(label) + '-' + name.replace("/", "_")
    newName = newName[:-4]
    feature = get_image_feature(opts, img_path, dictionary)
    np.save(join(out_dir, 'features', newName), feature)

def build_recognition_system(opts, n_worker=1):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)

    dictionary = np.load(join(out_dir, 'dictionary.npy'))

    # create partial function to represent extract_single_image_feature
    # since pool.map only takes one function as input
    training_files_plus_label = zip(train_files, train_labels)
    pFunc = partial(extract_single_image_features, opts, dictionary)
    with Pool(processes=n_worker) as pool:
        pool.map(pFunc, training_files_plus_label)


    # build labels array and feature matrix to pass to trained_system
    labels = np.ndarray(0)
    features = np.ndarray((0, opts.K * (pow(4, opts.L) - 1)//3))
    for f in os.listdir(join(out_dir, 'features')):
        label = int(f[0])
        labels = np.append(labels, label)
        loc = join(out_dir, 'features', f)
        a = np.load(loc)
        features = np.vstack((features, a))

    #example code snippet to save the learned system
    np.savez_compressed(join(out_dir, 'trained_system.npz'),
        features=features,
        labels=train_labels,
        dictionary=dictionary,
        SPM_layer_num=SPM_layer_num,
    )

def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''

    return (1 - np.sum(np.minimum(histograms, word_hist), axis=1))

def eval_(opts, dictionary, trained_features, trained_labels, labeledImage):

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    # extract individual name and label, extract features 
    (name, label) = labeledImage
    img_path = join(data_dir, name)
    feature = get_image_feature(opts, img_path, dictionary)
    
    dists = distance_to_set(feature, trained_features)
    assignedLabel = trained_labels[np.argmin(dists)]

    newName = name.replace("/", "_")
    newName = newName[:-4]
    np.save(join(out_dir, 'results', newName), [label, assignedLabel])
    
def evaluate_recognition_system(opts, n_worker=1):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, 'trained_system.npz'))
    dictionary = trained_system['dictionary']
    
    # load in features and labels from trained_system.npz
    trained_features = trained_system['features']
    trained_labels = trained_system['labels']

    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system['SPM_layer_num']

    test_files = open(join(data_dir, 'test_files.txt')).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, 'test_labels.txt'), np.int32)

    # create partial function to represent eval_
    # since pool.map only takes one function as input
    testing_files = zip(test_files, test_labels)
    pFunc = partial(eval_, test_opts, dictionary, trained_features, trained_labels)
    with Pool(processes=n_worker) as pool:
        pool.map(pFunc, testing_files)
    
    conf = np.zeros((8,8))
    # iterate through all resuls files 
    # x - which class the img is
    # y - what was predicted by the system
    for f in os.listdir(join(out_dir, 'results')):
        res_path = join(out_dir, 'results', f)
        a = np.load(res_path)
        x, y = a[0], a[1]
        # add 1 to appropriate confusion matrix index
        conf[x,y] += 1

    acc = np.trace(conf)/np.sum(conf) 
    return conf, acc
