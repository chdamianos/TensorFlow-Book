import tensorflow as tf
import numpy as np
import datetime as dt

def get_batch(X, size):
    a = np.random.choice(len(X), size, replace=False)
    return X[a]

class Autoencoder:
    def __init__(self, input_dim, hidden_dim, epoch=1000, batch_size=50, learning_rate=0.001):
        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        x = tf.placeholder(dtype=tf.float32, shape=[None, input_dim])
        with tf.name_scope('encode'):
            weights = tf.Variable(tf.random_normal([input_dim, hidden_dim], dtype=tf.float32), name='weights')
            biases = tf.Variable(tf.zeros([hidden_dim]), name='biases')
            encoded = tf.nn.sigmoid(tf.matmul(x, weights) + biases)
        with tf.name_scope('decode'):
            weights = tf.Variable(tf.random_normal([hidden_dim, input_dim], dtype=tf.float32), name='weights')
            biases = tf.Variable(tf.zeros([input_dim]), name='biases')
            decoded = tf.matmul(encoded, weights) + biases

        self.x = x
        self.encoded = encoded
        self.decoded = decoded

        self.loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.x, self.decoded))))
        self.loss_summ = tf.summary.scalar('loss', self.loss)
        self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
        
        self.saver = tf.train.Saver()

    def train(self, data):
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
                    batch_data = get_batch(data, self.batch_size)
                    l, _, l_summ = sess.run([self.loss, self.train_op, self.loss_summ], feed_dict={self.x: batch_data})
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

    def classify(self, data, labels):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.saver.restore(sess, './model.ckpt')
            # hidden are the embeddings, the reconstructed is the reconstructed data from input data -> self.x: data
            hidden, reconstructed, loss = sess.run([self.encoded, self.decoded, self.loss], feed_dict={self.x: data})
            # reconstructed = reconstructed[0]
            print('data', np.shape(data))
            print('reconstructed', np.shape(reconstructed))
            sumSquaredDiffs = np.sqrt(np.mean(np.square(data - reconstructed), axis=1))
            print('loss', np.shape(sumSquaredDiffs))
            horse_indices = np.where(labels == 7)[0]
            not_horse_indices = np.where(labels != 7)[0]
            horse_ssd = np.mean(sumSquaredDiffs[horse_indices])
            not_horse_ssd = np.mean(sumSquaredDiffs[not_horse_indices])
            print('horse SSD', horse_ssd)
            print('not horse SSD', not_horse_ssd)
            return hidden

    def decode(self, encoding):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.saver.restore(sess, './model.ckpt')
            reconstructed = sess.run(self.decoded, feed_dict={self.encoded: encoding})
        img = np.reshape(reconstructed, (32, 32))
        return img
