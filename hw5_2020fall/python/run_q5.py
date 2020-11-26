import numpy as np
import scipy.io
from nn import *
from collections import Counter
from util import relu
import matplotlib.pyplot as plt


train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

x_shape = train_x.shape[1]
max_iters = 100
# pick a batch size, learning rate
batch_size = 36 
learning_rate =  3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

# Q5.1 & Q5.2
# initialize layers here
initialize_weights(x_shape,hidden_size,params,'layer1')
initialize_weights(hidden_size,hidden_size,params,'layer2')
initialize_weights(hidden_size,hidden_size,params,'layer3')
initialize_weights(hidden_size,x_shape,params,'output')

#should look like your previous training loops
loss_arr_validation = []
itr_arr = []
for itr in range(max_iters):
    total_loss = 0
    for xb,_ in batches: 
        
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions
       
        # forward input
        h1 = forward(xb,params,'layer1', relu)
        h2 = forward(h1,params,'layer2', relu)
        h3 = forward(h2,params,'layer3', relu)
        probs = forward(h3,params,'output', sigmoid)

        # loss
        # be sure to add loss and accuracy to epoch totals 
        loss = np.sum((xb - probs) ** 2) # fix loss function 
        total_loss += loss

        # backward
        delta1 = 2 * (probs - xb) # if not working switch
        delta2 = backwards(delta1,params,'output',sigmoid_deriv) 
        delta3 = backwards(delta2,params,'layer3',relu_deriv)
        delta4 = backwards(delta3,params,'layer2',relu_deriv)
        backwards(delta4,params,'layer1',relu_deriv)

        # apply gradient
        for k,v in sorted(list(params.items())):
            if 'grad' in k:
                name = k.split('_')[1]
                params['m_'+name] = 0.9 * params['m_'+name] - learning_rate * params[k] 
                params[name]  = params[name] + params['m_'+name]

    total_loss = total_loss / len(batches)
    loss_arr_validation.append(total_loss)
    itr_arr.append(itr)

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9

plt.plot(itr_arr, loss_arr_validation)
plt.title("Autoencoder Loss - Training Data " + ", BatchSize = " + str(batch_size) + ", Init LR = " + str(3e-5))
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(['Training Data'], loc = 'upper left')
plt.show()

# Q5.3.2 & # Q5.3.1
import matplotlib.pyplot as plt        
from skimage.measure import compare_psnr as psnr
# evaluate PSNR
h1 = forward(valid_x,params,'layer1', relu)
h2 = forward(h1,params,'layer2', relu)
h3 = forward(h2,params,'layer3', relu)
probs = forward(h3,params,'output', sigmoid)

sum_psnr = 0

for i in range(len(valid_x)):
 
    img = valid_x[i]
    img = img.reshape(32,32)
    probs_img = probs[i].reshape(32,32)
    sum_psnr += psnr(img, probs_img)

    if i == 99 or i == 399 or i == 3339 or i == 1234 or i == 2600:
        
        plt.subplot(1,2,1)
        plt.imshow(img.reshape(32,32))
        plt.title("Image " + str(i) + " original")

        plt.subplot(1,2,2)
        plt.imshow(probs_img.reshape(32,32))
        plt.title("Image " + str(i) + " guess")
        plt.show()

avg_psnr = sum_psnr/len(valid_x)
print(avg_psnr)
