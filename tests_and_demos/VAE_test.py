import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data')

tf.reset_default_graph()

batch_size = 64

# globals, put in class?
dec_in_channels = 1
n_latent = 8

reshape_dim = [-1, 7, 7, dec_in_channels]
# Fucking python3 int division
inputs_decoder = 49 * dec_in_channels // 2

# data?
X_in = tf.placeholder(dtype = tf.float32, shape = [None, 28, 28], name = 'X')
Y    = tf.placeholder(dtype = tf.float32, shape = [None, 28, 28], name = 'Y')

Y_flat = tf.reshape(Y, shape = [-1, 28 * 28])
keep_prob = tf.placeholder(dtype = tf.float32, shape = (), name = 'keep_prob')


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
# inputs_decoder is global
        x = tf.layers.dense(sampled_z, units = inputs_decoder, activation = lrelu)
        x = tf.layers.dense(x, units = inputs_decoder * 2 + 1, activation = lrelu)
        x = tf.reshape(x, reshape_dim)
        x = tf.layers.conv2d_transpose(x, filters = 64, kernel_size = 4, 
                                       strides = 2, padding = 'same', activation = tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters = 64, kernel_size = 4, 
                                       strides = 1, padding = 'same', activation = tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters = 64, kernel_size = 4, 
                                       strides = 1, padding = 'same', activation = tf.nn.relu)

        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, units = 28 * 28, activation = tf.nn.sigmoid)
        img = tf.reshape(x, shape = [-1, 28, 28])
        return img


# Connection between encoder and decoder
sampled, mn, sd = encoder(X_in, keep_prob)
dec = decoder(sampled, keep_prob)


# Reconstruction loss
unreshaped = tf.reshape(dec, [-1, 28*28])
img_loss = tf.reduce_sum(tf.squared_difference(unreshaped, Y_flat), 1)
# Don't quite see what this does?
latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), 1)
loss = tf.reduce_mean(img_loss + latent_loss)
# What is the adam optimizer? magic number here?
optimizer = tf.train.AdamOptimizer(0.0005).minimize(loss)
inter_op_parallelism_threads = 1
intra_op_parallelism_threads = 1
sess = tf.Session(config = tf.ConfigProto(inter_op_parallelism_threads = inter_op_parallelism_threads,
                                          intra_op_parallelism_threads = intra_op_parallelism_threads))
sess.run(tf.global_variables_initializer())


# training part, should be a function
iterations = 30000
for i in range(iterations):
    batch = [np.reshape(b, [28, 28]) for b in mnist.train.next_batch(batch_size = batch_size)[0]]
    sess.run(optimizer, feed_dict = {X_in: batch, Y: batch, keep_prob: 0.8})
    # magic number?
    if not i % 200:
        ls, d, i_ls, d_ls, mu, sigm = sess.run([loss, dec, img_loss, latent_loss, mn, sd], 
                                                feed_dict = {X_in: batch, Y: batch, keep_prob: 1.0})
        plt.imshow(np.reshape(batch[0], [28, 28]), cmap = 'gray')
        plt.show()
        plt.imshow(d[0], cmap = 'gray')
        plt.show()
        print(i, ls, np.mean(i_ls), np.mean(d_ls))
