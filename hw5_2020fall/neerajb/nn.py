import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

############################## Q 2.1 ##############################
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    
    lim = np.sqrt(6)/np.sqrt(in_size + out_size)

    W = np.random.uniform(-lim, lim, (in_size,out_size))
    b = np.zeros(out_size)
        
    params['W' + name] = W
    params['b' + name] = b

############################## Q 2.2.1 ##############################
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    
    
    res = 1/(1 + np.exp(-x))

    return res

############################## Q 2.2.1 ##############################
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]

    pre_act = np.dot(X, W) + b
    post_act = np.zeros(pre_act.shape)
    for i in range(len(pre_act)):
        post_act[i] = activation(pre_act[i])
   
    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

############################## Q 2.2.2  ##############################
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    #https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
    
    e_x = np.exp(x-np.max(x))
    res = e_x/np.sum(e_x)
 
    return res

############################## Q 2.2.3 ##############################
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):

    loss = 0.0
    
    for i in range(len(y)):
        loss -= np.dot(y[i],np.log(probs[i]))
    max_index_probs = np.argmax(probs, axis = 1)
    max_index_y = np.argmax(y, axis = 1)

    diff = max_index_probs - max_index_y

    N = np.count_nonzero(diff)

    acc = 1 - N/len(diff)
    
    return loss, acc 

############################## Q 2.3 ##############################
# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):

    # """
    # Do a backwards pass

    # Keyword arguments:
    # delta -- errors to backprop
    # params -- a dictionary containing parameters
    # name -- name of the layer
    # activation_deriv -- the derivative of the activation_func
    # """
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]

    dy_dx = W 
    dy_dW = X
    dy_db = np.zeros((1,X.shape[0])) + 1
    
    dl_dx = np.dot(delta * activation_deriv(post_act), dy_dx.T)
    dl_dw = np.dot((delta * activation_deriv(post_act)).T, dy_dW).T
    dl_db = np.dot((delta * activation_deriv(post_act)).T, dy_db.T).reshape(b.shape)

    grad_W = dl_dw
    grad_X = dl_dx
    grad_b = dl_db

    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b

    assert(grad_W.shape == W.shape)
    assert(grad_b.shape == b.shape)
    assert(grad_X.shape == X.shape)

    return grad_X

############################## Q 2.4 ##############################
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    batches = []

    index_list = np.arange(len(x))
    np.random.shuffle(index_list)

    x = x[index_list]
    y = y[index_list]
    
    num_batches = int(len(x)/batch_size)

    for i in range(0, int(len(x)), batch_size):
        batchx = x[i:i+batch_size]
        batchy = y[i:i+batch_size]

        batches.append((batchx, batchy))
    
    return batches
