import tensorflow as tf
import numpy as np
import datetime as dt


def get_batch(X, Xn, size):
    a = np.random.choice(len(X), size, replace=False)
    return X[a], Xn[a]

# A denoising autoencoder is pretty much the same architecture as a normal autoencoder. The input is noised up,
# and cost function tries to denoise it by minimizing the construction error from denoised input to clean output.

class Autoencoder:
    def __init__(self, input_dim, hidden_dim, epoch=1000, batch_size=50, learning_rate=0.001):
        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.x = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='x')
        self.x_noised = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='x_noised')
        with tf.name_scope('encode'):
            weights = tf.Variable(tf.random_normal([input_dim, hidden_dim], dtype=tf.float32), name='weights')
            biases = tf.Variable(tf.zeros([hidden_dim]), name='biases')
            encoded = tf.nn.sigmoid(tf.matmul(self.x_noised, weights) + biases, name='encoded')
        with tf.name_scope('decode'):
            weights = tf.Variable(tf.random_normal([hidden_dim, input_dim], dtype=tf.float32), name='weights')
            biases = tf.Variable(tf.zeros([input_dim]), name='biases')
            decoded = tf.matmul(encoded, weights) + biases

        self.encoded = encoded
        self.decoded = decoded

        self.loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.x, self.decoded))))
        self.loss_summ = tf.summary.scalar('loss', self.loss)
        self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

        self.saver = tf.train.Saver()

    def add_noise(self, data):
        noise_type = 'mask-0.2'
        if noise_type == 'gaussian':
            n = np.random.normal(0, 0.1, np.shape(data))
            return data + n
        if 'mask' in noise_type:
            frac = float(noise_type.split('-')[1])
            temp = np.copy(data)
            for i in temp:
                n = np.random.choice(len(i), round(frac * len(i)), replace=False)
                i[n] = 0
            return temp

    def train(self, data):
        data_noised = self.add_noise(data)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            # Create a directory everytime so that tensorboard displays each job individually
            now = dt.datetime.now()
            currentDir = "./logs/" + now.strftime("%Y%m%d-%H%M%S") + "/"
            writer = tf.summary.FileWriter(currentDir, graph=sess.graph)
            tf.train.write_graph(sess.graph_def, "./logs/", 'graph.pbtxt')
            for i in range(self.epoch):
                for j in range(np.shape(data)[0] // self.batch_size):
                    batch_data, batch_data_noised = get_batch(data, data_noised, self.batch_size)
                    l, _, l_summ = sess.run([self.loss, self.train_op, self.loss_summ], feed_dict={self.x: batch_data,
                                                                                    self.x_noised: batch_data_noised})
                if i % 100 == 0:
                    print('epoch {0}: loss = {1}'.format(i, l))
                    self.saver.save(sess, './model.ckpt')
                writer.add_summary(summary=l_summ, global_step=i)
                writer.flush()
            self.saver.save(sess, './model.ckpt')
            writer.close()

    def test(self, data):
        with tf.Session() as sess:
            self.saver.restore(sess, './model.ckpt')
            hidden, reconstructed = sess.run([self.encoded, self.decoded], feed_dict={self.x: data})
        print('input', data)
        print('compressed', hidden)
        print('reconstructed', reconstructed)
        return reconstructed

    def get_params(self):
        with tf.Session() as sess:
            self.saver.restore(sess, './model.ckpt')
            weights, biases = sess.run([self.weights1, self.biases1])
        return weights, biases

    def encode(self, data):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.saver.restore(sess, './model.ckpt')
            # hidden are the embeddings
            hidden = sess.run(self.encoded, feed_dict={self.x: data, self.x_noised: data})
            return hidden

    def decode(self, encoding):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.saver.restore(sess, './model.ckpt')
            reconstructed = sess.run(self.decoded, feed_dict={self.encoded: encoding})
        img = np.reshape(reconstructed, (32, 32))
        return img
