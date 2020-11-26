import numpy as np
import scipy.io
from nn import *

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

x_shape = train_x.shape[1]
y_shape = train_y.shape[1]

max_iters = 50
# pick a batch size, learning rate
batch_size = 15
# best 0.3e-2
learning_rate = 0.3e-2 
hidden_size = 64
##########################
##### your code here #####
##########################

batches_training = get_random_batches(train_x,train_y,batch_size)
batches_validation = get_random_batches(valid_x,valid_y,batch_size)
batch_num = len(batches_training)

params = {}

# initialize layers here
##########################
##### your code here #####
##########################

initialize_weights(x_shape,hidden_size,params,'layer1')
initialize_weights(hidden_size,y_shape,params,'output')

init_weights_input = params['Wlayer1']

# with default settings, you should get loss < 150 and accuracy > 80%
acc_arr_training = []
loss_arr_training = []

acc_arr_validation = []
loss_arr_validation = []

itr_arr = []
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    valid_loss = 0
    valid_acc = 0
    # iterate over training batches
    for xb,yb in batches_training:
        # forward
        h1 = forward(xb,params,'layer1')
        probs = forward(h1,params,'output',softmax)

        # loss
        # be sure to add loss and accuracy to epoch totals 
        loss, acc = compute_loss_and_acc(yb, probs)
        total_loss += loss
        total_acc += acc

        # backward
        delta1 = probs - yb
        delta2 = backwards(delta1,params,'output',linear_deriv)
        backwards(delta2,params,'layer1',sigmoid_deriv)

        # apply gradient
        for k,v in sorted(list(params.items())):
            if 'grad' in k:
                name = k.split('_')[1]
                params[name]  = params[name] - learning_rate * params[k]

    total_acc = total_acc / len(batches_training)
    total_loss = total_loss / len(batches_training)
    acc_arr_training.append(total_acc)
    loss_arr_training.append(total_loss)
    itr_arr.append(itr)

    if itr % 2 == 0:
        print("train: itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))

    # forward prop through validation batches, not updating weights
    for xb,yb in batches_validation:
        # forward
        h1 = forward(xb,params,'layer1')
        probs = forward(h1,params,'output',softmax)

        # loss
        # be sure to add loss and accuracy to epoch totals 
        loss, acc = compute_loss_and_acc(yb, probs)
        valid_loss += loss
        valid_acc += acc
    
    valid_acc = valid_acc / len(batches_validation)
    valid_loss = valid_loss / len(batches_validation)
    acc_arr_validation.append(valid_acc)
    loss_arr_validation.append(valid_loss)
    if itr % 2 == 0:
        print("valid: itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,valid_loss,valid_acc))

print('Validation accuracy: ',valid_acc)
if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        plt.show()
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.2
# run on validation set and report accuracy! should be above 75%
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
# plot accuracy and loss 
plt.plot(itr_arr, acc_arr_training)
plt.plot(itr_arr, acc_arr_validation)
plt.title("Accuracy - Training vs. Validation Data " + ", BatchSize = " + str(batch_size) + ", LR = " + str(learning_rate))
plt.xlabel("Epochs")
plt.ylabel("Accuracy %")
plt.legend(['Training Data', 'Validation Data'], loc = 'upper left')
plt.show()
plt.plot(itr_arr, loss_arr_training)
plt.plot(itr_arr, loss_arr_validation)
plt.title("Loss - Training vs. Validation Data " + ", BatchSize = " + str(batch_size) + ", LR = " + str(learning_rate))
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(['Training Data', 'Validation Data'], loc = 'upper left')
plt.show()

# Q3.3
# visualize weights here

# post_weights_input = params['Wlayer1']

# fig = plt.figure(1, (4., 4.))
# grid = ImageGrid(fig, 111,  # similar to subplot(111)
#                  nrows_ncols=(1, 2),  # creates 2x2 grid of axes
#                  axes_pad=0.1,  # pad between axes in inch.
#                  )

# grid[0].imshow(init_weights_input.reshape(256,256))
# grid[1].imshow(post_weights_input.reshape(256,256))  # The AxesGrid object work as a list of axes.

# plt.show()
########################
w = params['Wlayer1']
ind = np.array([w[:, i] for i in range(w.shape[1])])
fig2 = plt.figure()
grid = ImageGrid(fig2, 211,  # similar to subplot(111)
                 nrows_ncols=(8, 8),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )
for ax, im in zip(grid, ind):
    im = im.reshape((32,32))
    ax.imshow(im)

plt.show()  # The AxesGrid object work as a list of axes.
##########################

########################
w = init_weights_input
ind = np.array([w[:, i] for i in range(w.shape[1])])
fig2 = plt.figure()
grid = ImageGrid(fig2, 211,  # similar to subplot(111)
                 nrows_ncols=(8, 8),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )
for ax, im in zip(grid, ind):
    im = im.reshape((32,32))
    ax.imshow(im)

plt.show()  # The AxesGrid object work as a list of axes.
##########################

# Q3.4  
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))

h1 = forward(train_x,params,'layer1')
probs = forward(h1,params,'output',softmax)

for i in range(len(train_y)):
    label_valid = train_y[i,:]
    index_valid = np.argmax(label_valid)

    label_guess = probs[i,:]
    index_guess = np.argmax(label_guess)

    confusion_matrix[index_valid, index_guess] += 1

import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()