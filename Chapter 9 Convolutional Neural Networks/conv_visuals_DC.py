import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cifar_tools_DC
import tensorflow as tf

# use this to suppress plots
# matplotlib.use('agg')

names, data, labels = \
    cifar_tools_DC.read_data(
        './cifar-10-batches-py')


def show_conv_results(data, filename=None):
    fig, axes = plt.subplots(nrows=4, ncols=8)
    ax = axes.ravel()
    for i in range(np.shape(data)[3]):
        img = data[0, :, :, i]
        ax[i].imshow(img, cmap='Greys_r', interpolation='none')
        ax[i].axis('off')
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def show_weights(W, filename=None):
    fig, axes = plt.subplots(nrows=4, ncols=8)
    # Define just enough rows and columns to show the 32 results from the convolution
    ax=axes.ravel()
    # visualise each filter, note that the number of filters is in the 4th dimension
    # so for a filter with dimensions [5,5,1,32] we have 32 filters
    for i in range(np.shape(W)[3]):
        img = W[:, :, 0, i]
        ax[i].imshow(img, cmap='Greys_r', interpolation='none')
        ax[i].axis('off')
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


raw_data = data[4, :]
raw_img = np.reshape(raw_data, (24, 24))
fig, ax= plt.subplots()
ax.imshow(raw_img, cmap='Greys_r')
plt.show()

# The input to the conv2d needs to be as [batch, in_height, in_width, in_channels]
# in the case of a single image the batch=1, in_height=24, in_width=24, in_channels=1 (since it's greyscale)
x = tf.reshape(raw_data, shape=[-1, 24, 24, 1])
# Define the tensor representing the random filters
# 5,5 -> size of filer on image, 1-> input dimension (1 for grayscale, 3 for RGB)
# 32 -> number of convolutions
W = tf.Variable(tf.random_normal([5, 5, 1, 32]))
b = tf.Variable(tf.random_normal([32]))

conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
conv_with_b = tf.nn.bias_add(conv, b)
conv_out = tf.nn.relu(conv_with_b)

# After a convolution layer extracts useful features, it’s usually a good idea to reduce the size
# of the convolved outputs. Rescaling or subsampling a convolved output helps reduce the
# number of parameters, which in turn can help to not overfit the data.

# max-pooling: sweeps a window acrossan image and picks the pixel with the maximum value.
# Depending on the stride-length, the resulting image is a fraction of the size of the original.
# This is useful because it lessens the dimensionality of the data, consequently lowering the number
# of parameters in future steps.
k = 2
# max-pooling function with k=2 halves the image size, and produces lower-resolution convolved outputs,
maxpool = tf.nn.max_pool(conv_out, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    W_val = sess.run(W)
    show_weights(W_val)

    xVal = sess.run(x)
    fig1, ax1 = plt.subplots()
    ax1.imshow(xVal.squeeze(), cmap='Greys_r', interpolation='none')

    conv_val = sess.run(conv)
    show_conv_results(conv_val)
    print(np.shape(conv_val))

    conv_out_val = sess.run(conv_out)
    show_conv_results(conv_out_val)
    print(np.shape(conv_out_val))

    maxpool_val = sess.run(maxpool)
    show_conv_results(maxpool_val)
    print(np.shape(maxpool_val))
