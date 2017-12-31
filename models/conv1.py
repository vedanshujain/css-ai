from parser.encoder import Encoder
from parser.reader import Reader
import tensorflow as tf
import numpy as np


class Conv1:

    def __init__(self, prop_count, values_count, patch_size):
        self.prop_count = prop_count
        self.values_count = values_count
        self.X = tf.placeholder(tf.float32, shape=(None, values_count, prop_count, patch_size))
        self.Y = tf.placeholder(tf.float32, shape=(None, prop_count, values_count))
        self.cnn_model = None

    def model(self):

        if self.cnn_model is not None:
            return

        # Conv 1
        model = tf.layers.conv2d(inputs=self.X, filters=200, padding='same', activation=tf.nn.relu,
                                 kernel_size=3, name='conv1')

        # Conv 2
        model = tf.layers.conv2d(inputs=model, strides=[3, 3], filters=100, padding='same', activation=tf.nn.relu,
                                 kernel_size=3, name='conv2')

        # Conv 3
        # model = tf.layers.conv2d(inputs=model, strides=[3, 3], filters=300, padding='same', activation=tf.nn.relu,
        #                          kernel_size=3, name='conv3')

        # pool 1
        model = tf.layers.max_pooling2d(inputs=model, pool_size=2, strides=2, name='max1')

        # Conv 4
        model = tf.layers.conv2d(inputs=model, strides=[2, 2], filters=100, padding='same', activation=tf.nn.relu,
                                 kernel_size=2, name='conv4')

        # pool 2
        model = tf.layers.max_pooling2d(inputs=model, pool_size=2, strides=1, name='max2')

        model = tf.layers.flatten(model)

        # dense
        # model = tf.layers.dense(inputs=model, units=self.prop_count*self.values_count, activation=tf.nn.relu, name='fc1')
        model = tf.layers.dense(inputs=model, units=self.prop_count*self.values_count, activation=tf.nn.relu, name='fc2')

        model = tf.reshape(model, (-1, self.prop_count, self.values_count))

        # normalizing probabilities
        model = tf.nn.softmax(model)

        self.cnn_model = model

        return model

    def loss(self):
        if self.cnn_model is None:
            raise Exception("Variable cnn_model is not initialised yet")

        input_count = tf.cast(tf.shape(self.X)[0], tf.float32)

        return tf.reduce_sum(
           tf.square(tf.subtract(self.Y, self.cnn_model))
        ) / (input_count * 2)

    def train(self):
        self.model()
        loss = self.loss()
        optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
        return optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    def fill_feed_dict(self, X, Y):
        return {self.X: np.swapaxes(X, 1, 3), self.Y: Y}

    def eval(self):
        return self.model()

