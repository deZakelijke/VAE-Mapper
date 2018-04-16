import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data')

tf.reset_default_graph()

batch_size = 64

X_in = tf.placeholder(dtype = tf.float32, shape = [None, 28, 28], name = 'X')
Y    = tf.placeholder(dtype = tf.float32, shape = [None, 28, 28], name = 'Y')

Y_flat = tf.reshape(Y, shape = [-1, 28 * 28])
keep_prob = tf.placeholder(dtype = tf.float32, shape = (), name = 'keep_prob')

dec_in_channels = 1
n_latent = 8

reshape_dim = [-1, 7, 7, dec_in_channels]
inputs_decoder = 49 * dec_in_channels / 2

# leaky relu is not in tensorflow
def lrelu(x, alpha = 0.3):
    return tf.maximum(x, tf.multiply(x, alpha))


# Encoder function
def encoder(X_in, keep_prob):
    with tf.variable_scope("encoder", reuse = None):
        X = tf.reshape(X_in, shape = [-1, 28, 28, 1])
        x = tf.layers.conv2d(X, filters = 64, kernel_size = 4, strides = 2, 
                             padding = 'same', activation = lrelu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters = 64, kernel_size = 4, strides = 2, 
                             padding = 'same', activation = lrelu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters = 64, kernel_size = 4, strides = 1, 
                             padding = 'same', activation = lrelu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.contrib.layers.flatten(x)
# n_latent is global, so not defined in this function
        mn = tf.layers.dense(x, units = n_latent)
        sd = 0.5 * tf.layers.dense(x, units = n_latent)
        epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], n_latent]))
        z = mn + tf.multiply(epsilon, tf.exp(sd))
        return z, mn, sd

# Decoder function
def decoder(sampled_z, keep_prob):
    with tf.variable_scope("decoder", reuse = None):
        x = tf.layers.dense(sampled_z, units = inputs_decoder, activation = lrelu)
        x = tf.layers.dense(x, units = inputs_decoder * 2 + 1, activation = lrelu)
        x = tf.reshape(x, reshaped_dim)
        x = tf.layers.conv2d_transpose(x, filters = 64, kernel_size = 4, 
                                       strides = 2, padding = 'same', activation = tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters = 64, kernel_size = 4, 
                                       strides = 1, padding = 'same', activation = tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters = 64, kernel_size = 4, 
                                       strides = 1, padding = 'same', activation = tf.nn.relu)

        x = tf.contrib.layers.flatten(x)
        x = tf.layyers.dense(x, units = 28 * 28, activation = tf.nn.sigmoid)
        img = tf.reshape(x, shape = [-1, 28, 28])
        return img
