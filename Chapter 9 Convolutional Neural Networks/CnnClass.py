import tensorflow as tf
import datetime as dt


class Cnn:
    def __init__(self, input_dim, output_dim, epoch=1000, batch_size=100, learning_rate=0.001):
        # MODEL PARAMETERS
        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.output_dim = output_dim
        # INPUT PLACEHOLDERS
        # x holds the input images
        self.x = tf.placeholder(tf.float32, [None, self.input_dim * self.input_dim], name='inputPlaceholder')
        # y holds the output targets
        self.y = tf.placeholder(tf.float32, [None, self.output_dim], name='targetPlaceholder')
        # TRAINABLE VARIABLES
        # 5x5 filter to produce 64 different arrays
        with tf.name_scope('ConvolutionVariables'):
            self.W1 = tf.Variable(tf.random_normal([5, 5, 1, 64]), name='W1')
            self.b1 = tf.Variable(tf.random_normal([64]), name='b1')
            # 5x5 filter to produce again 64 different images
            # note that dim=2 is 64 to accommodate the 64 different images
            # produced from W1, b1
            self.W2 = tf.Variable(tf.random_normal([5, 5, 64, 64]), name='W2')
            self.b2 = tf.Variable(tf.random_normal([64]), name='b2')
        with tf.name_scope('FullyConnectedVariables'):
            # Fully connected layer
            self.W3 = tf.Variable(tf.random_normal([6 * 6 * 64, 1024]), name='W3')
            self.b3 = tf.Variable(tf.random_normal([1024]), name='b3')
        with tf.name_scope('OutputVariables'):
            # Output layers
            self.W_out = tf.Variable(tf.random_normal([1024, self.output_dim]), name='W_out')
            self.b_out = tf.Variable(tf.random_normal([self.output_dim]), name='W_out')
        # OPTIMIZATION
        self.train_op, self.modelOut, self.cost = self.optimization()
        # MODEL ASSESSMENT
        self.correct_pred = tf.equal(tf.argmax(self.modelOut, 1), tf.argmax(self.y, 1), name='correct_pred')
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32), name='accuracy')
        # PARAMETER SUMMARIES
        self.summaries_tensor = self.model_summaries()
        # MODEL SAVE
        self.saver = tf.train.Saver()

    def optimization(self):
        # OPTIMIZATION
        modelOut = self.model()
        modelOut = tf.identity(modelOut, name="modelOut")
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=modelOut, labels=self.y),
                              name='cost')
        train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)
        return train_op, modelOut, cost

    def model_summaries(self):
        # PARAMETER SUMMARIES
        cost_summary = tf.summary.scalar('cost_summary', self.cost)
        W1_summary = tf.summary.histogram('W1', self.W1)
        x_summary = tf.summary.image('x', self.x_reshaped)
        summaries_tensor = tf.summary.merge([cost_summary, W1_summary, x_summary])
        return summaries_tensor

    def conv_layer(self, data, W, b):
        conv = tf.nn.conv2d(data, W, strides=[1, 1, 1, 1], padding='SAME', name='conv')
        conv_with_b = tf.nn.bias_add(conv, b, name='conv_with_b')
        conv_out = tf.nn.relu(conv_with_b, name='conv_out')
        return conv_out

    def maxpool_layer(self, conv, k=2):
        return tf.nn.max_pool(conv, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name='maxpool_layer')

    def model(self):
        self.x_reshaped = tf.reshape(self.x, shape=[-1, 24, 24, 1])
        with tf.name_scope('Convolution'):
            # Construct the first layer of convolution and maxpooling
            conv_out1 = self.conv_layer(self.x_reshaped, self.W1, self.b1)
            self.maxpool_out1 = self.maxpool_layer(conv_out1)
            norm1 = tf.nn.lrn(self.maxpool_out1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
            # Construct the second layer
            conv_out2 = self.conv_layer(norm1, self.W2, self.b2)
            norm2 = tf.nn.lrn(conv_out2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
            maxpool_out2 = self.maxpool_layer(norm2)
        with tf.name_scope('FullyConnected'):
            # Lastly, construct the concluding fully connected layers
            maxpool_reshaped = tf.reshape(maxpool_out2, [-1, self.W3.get_shape().as_list()[0]], name='maxpool_reshaped')
            local = tf.add(tf.matmul(maxpool_reshaped, self.W3), self.b3, name='local')
            local_out = tf.nn.relu(local, name='local_out')
        with tf.name_scope('Output'):
            out = tf.add(tf.matmul(local_out, self.W_out), self.b_out, name='out')

        return out

    def train(self, data, labels, savedModelLocation):

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            # Create a directory everytime so that tensorboard displays each job individually
            now = dt.datetime.now()
            currentDir = "./logs/" + now.strftime("%Y%m%d-%H%M%S") + "/"
            writer = tf.summary.FileWriter(currentDir, graph=sess.graph)
            onehot_labels = tf.one_hot(labels, self.output_dim, on_value=1., off_value=0., axis=-1)
            onehot_vals = sess.run(onehot_labels)
            globalStep = 0
            for j in range(0, self.epoch):
                print('EPOCH', j)
                for i in range(0, len(data), self.batch_size):
                    batch_data = data[i:i + self.batch_size, :]
                    batch_onehot_vals = onehot_vals[i:i + self.batch_size, :]
                    _, accuracy_val, var_summary = sess.run([self.train_op, self.accuracy, self.summaries_tensor],
                                                            feed_dict={self.x: batch_data, self.y: batch_onehot_vals})
                    if i % 1000 == 0:
                        print(i, accuracy_val)
                    writer.add_summary(summary=var_summary, global_step=globalStep)
                    writer.flush()
                    globalStep += 1
                print('DONE WITH EPOCH')
                self.saver.save(sess, savedModelLocation)

    def inference(self, dataPoint, savedModelLocation):
        with tf.Session() as sess:
            self.saver.restore(sess, savedModelLocation)
            result = sess.run([self.modelOut], feed_dict={self.x: dataPoint.reshape((-1, 576))})
        return result
