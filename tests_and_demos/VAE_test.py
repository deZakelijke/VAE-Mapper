import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

class VAE(object):

    def __init__(self, batch_size, dec_in_channels, n_latent):
        self.batch_size = batch_size
        self.dec_in_channels = dec_in_channels
        self.n_latent = n_latent
        self.reshape_dim = [-1, 7, 7, dec_in_channels]
        self.inputs_decoder = 49 * dec_in_channels // 2
        self.X_in = tf.placeholder(dtype = tf.float32, shape = [None, 28, 28], name = 'X')
        self.Y    = tf.placeholder(dtype = tf.float32, shape = [None, 28, 28], name = 'Y')
        self. Y_flat = tf.reshape(self.Y, shape = [-1, 28 * 28])
        self.keep_prob = tf.placeholder(dtype = tf.float32, shape = (), name = 'keep_prob')
        # Number of threads that run in parallel
        inter_op_parallelism_threads = 4
        # Number of threads used for singe operation that can be run concurrently
        intra_op_parallelism_threads = 4
        # Create Session and init somehting?
        self.sess = tf.Session(config = tf.ConfigProto(
                               inter_op_parallelism_threads = inter_op_parallelism_threads,
                               intra_op_parallelism_threads = intra_op_parallelism_threads))
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        # Connection between encoder and decoder
        # Not sure if it should be here
        self.sampled, self.mn, self.sd = self.encoder(self.X_in)
        self.dec = self.decoder(self.sampled)
        
        # Reconstruction loss
        unreshaped = tf.reshape(self.dec, [-1, 28*28])
        self.img_loss = tf.reduce_sum(tf.squared_difference(unreshaped, self.Y_flat), 1)
        # Don't quite see what this does?
        self.latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * self.sd - tf.square(self.mn) - tf.exp(2.0 * self.sd), 1)
        self.loss = tf.reduce_mean(self.img_loss + self.latent_loss)
        # What is the adam optimizer? magic number here?
        # Should this be done at init?
        # If so, should the loss be redefined here or made into a class variable        
        # Let's try
        self.optimizer = tf.train.AdamOptimizer(0.0005).minimize(self.loss)
        



    # leaky relu is not in tensorflow
    def lrelu(self, x, alpha = 0.3):
        return tf.maximum(x, tf.multiply(x, alpha))
    
    
    # Encoder function
    def encoder(self, X_in):
        with tf.variable_scope("encoder", reuse = None):
            X = tf.reshape(X_in, shape = [-1, 28, 28, 1])
            x = tf.layers.conv2d(X, filters = 64, kernel_size = 4, strides = 2, 
                                 padding = 'same', activation = self.lrelu)
            x = tf.nn.dropout(x, self.keep_prob)
            x = tf.layers.conv2d(x, filters = 64, kernel_size = 4, strides = 2, 
                                 padding = 'same', activation = self.lrelu)
            x = tf.nn.dropout(x, self.keep_prob)
            x = tf.layers.conv2d(x, filters = 64, kernel_size = 4, strides = 1, 
                                 padding = 'same', activation = self.lrelu)
            x = tf.nn.dropout(x, self.keep_prob)
            x = tf.contrib.layers.flatten(x)
            mn = tf.layers.dense(x, units = self.n_latent)
            sd = 0.5 * tf.layers.dense(x, units = self.n_latent)
            epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], self.n_latent]))
            z = mn + tf.multiply(epsilon, tf.exp(sd))
            return z, mn, sd
    
    # Decoder function
    def decoder(self, sampled_z):
        with tf.variable_scope("decoder", reuse = tf.AUTO_REUSE):
            x = tf.layers.dense(sampled_z, units = self.inputs_decoder, activation = self.lrelu)
            x = tf.layers.dense(x, units = self.inputs_decoder * 2 + 1, activation = self.lrelu)
            x = tf.reshape(x, self.reshape_dim)
            x = tf.layers.conv2d_transpose(x, filters = 64, kernel_size = 4, 
                                           strides = 2, padding = 'same', activation = tf.nn.relu)
            x = tf.nn.dropout(x, self.keep_prob)
            x = tf.layers.conv2d_transpose(x, filters = 64, kernel_size = 4, 
                                           strides = 1, padding = 'same', activation = tf.nn.relu)
            x = tf.nn.dropout(x, self.keep_prob)
            x = tf.layers.conv2d_transpose(x, filters = 64, kernel_size = 4, 
                                           strides = 1, padding = 'same', activation = tf.nn.relu)
    
            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(x, units = 28 * 28, activation = tf.nn.sigmoid)
            img = tf.reshape(x, shape = [-1, 28, 28])
            return img

    def train_model(self, iterations):
        for i in range(iterations):
            batch = [np.reshape(b, [28, 28]) for b in mnist.train.next_batch(batch_size = self.batch_size)[0]]
            self.sess.run(self.optimizer, feed_dict = {self.X_in: batch, self.Y: batch, self.keep_prob: 0.8})
            if not i % 200:
                ls, d, i_ls, d_ls, mu, sigm = self.sess.run([self.loss, self.dec, self.img_loss, 
                                                        self.latent_loss, self.mn, self.sd], 
                                                        feed_dict = {self.X_in: batch, self.Y: batch, self.keep_prob: 1.0})
                plt.imshow(np.reshape(batch[0], [28, 28]), cmap = 'gray')
                plt.show()
                plt.imshow(d[0], cmap = 'gray')
                plt.show()
                print(i, ls, np.mean(i_ls), np.mean(d_ls))


    def generate_new_data(self):
        # Seems like an ugly fix
        self.sess.run(tf.initialize_all_variables())
        randoms = [np.random.normal(0, 1, self.n_latent) for _ in range(10)]
        imgs = self.sess.run(self.dec, feed_dict = {self.sampled: randoms, self.keep_prob: 1.0})
        imgs = [np.reshape(imgs[i], [28, 28]) for i in range(len(imgs))]
        
        for img in imgs:
            print("a")
            plt.figure(figsize = (1, 1))
            plt.axis('off')
            plt.imshow(img, cmap = 'gray')


    def save_model(self, model_name, global_step):
        saver = tf.train.Saver()
        saver.save(self.sess, model_name, global_step = global_step)
        
    def load_model(self, model_name, model_dir, global_step):
        saver = tf.train.import_meta_graph("%s-%s.meta" %(model_dir + model_name, global_step))
        saver.restore(self.sess, tf.train.latest_checkpoint(model_dir))


if __name__ == '__main__':
    mnist = input_data.read_data_sets('MNIST_data')
    
    # What does this do?
    tf.reset_default_graph()

    batch_size = 64
    dec_in_channels = 1
    n_latent = 8
    iterations = 3000
    model_dir = "./model_dir/"
    model_name = "my_test_model"
    train = False
    gen = True

    vae = VAE(batch_size, dec_in_channels, n_latent)

    if train:
        vae.train_model(iterations)
        vae.save_model(model_dir + model_name, iterations)

    if gen:
        vae.load_model(model_name, model_dir, iterations)
        
        vae.generate_new_data()
    
