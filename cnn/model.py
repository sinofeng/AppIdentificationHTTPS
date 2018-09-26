import tensorflow as tf
import numpy as np

from sklearn.metrics import f1_score
from tensorflow.contrib import slim

class CNN():
    def __init__(self, epoch, batch_size):

        self.epoch = epoch
        self.batch_size = batch_size

        self.x = tf.placeholder(tf.float32, [None, 64, 64, 1], name='input_x')
        self.y = tf.placeholder(tf.int32, [None], name='input_y')

        self.init_graph()
        self.sess = self._init_session()
        self.sess.run(tf.global_variables_initializer())

    def init_graph(self):

        net = slim.conv2d(self.x, 64, [3, 3], 1, padding='SAME')
        net = slim.conv2d(net, 64, [3, 3], 1, padding='SAME')
        net = slim.max_pool2d(net, [2, 2], 2)

        net = tf.reshape(net, [-1, 32 * 32 * 64])
        net = tf.layers.dense(net, 256, activation=tf.nn.relu)
        net = tf.layers.dense(net, 256, activation=tf.nn.relu)

        self.output = tf.layers.dense(net, 3, activation=None)

        self.prediction = tf.argmax(self.output, axis=-1)


        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.output,
            labels=self.y
        ))

        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.loss)

    # shuffle four lists simutaneously
    def shuffle_in_unison_scary(self, a, b):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)

    def _init_session(self):
        config = tf.ConfigProto(device_count={"gpu": 0})
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)

    def fit_on_batch(self, x, y):


        feed_dict = {self.x : x,
                     self.y : y}

        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss

    def get_batch(self, x, y, batch_size, index):
        start = index * batch_size
        end = (index + 1) * batch_size
        end = end if end < len(y) else len(y)

        return x[start:end], y[start:end]



    def predict(self, x):
        dummy_y = [1] * len(x)
        batch_index = 0
        x_batch, y_batch = self.get_batch(x, dummy_y, self.batch_size, batch_index)

        y_pred = None
        while len(x_batch) > 0:
            num_batch = len(y_batch)
            feed_dict = {self.x : x_batch,
                         self.y : y_batch}

            batch_out = self.sess.run(self.prediction, feed_dict=feed_dict)

            if batch_index == 0:
                y_pred = np.reshape(batch_out, (num_batch,))
            else:
                y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch,))))
            batch_index += 1
            x_batch, y_batch = self.get_batch(x, dummy_y, self.batch_size, batch_index)

        return y_pred

    def evaluate(self, x,  y):

        y_pred = self.predict(x)
        return f1_score(y, y_pred, average='macro')


    def fit(self, x_train, y_train, x_val, y_val):
        for epoch in range(self.epoch):
            # self.shuffle_in_unison_scary(x_train, y_train)
            total_batch = int(len(y_train) / self.batch_size)
            for batch_index in range(total_batch):
                x_batch, y_batch = self.get_batch(x_train, y_train, self.batch_size, batch_index)
                self.fit_on_batch(x_batch, y_batch)

            train_result = self.evaluate(x_train, y_train)
            val_result = self.evaluate(x_val, y_val)

            print("[%d] train-result=%.4f, valid-result=%.4f " % (epoch + 1, train_result, val_result))