from parser.encoder import Encoder
from parser.reader import Reader
import tensorflow as tf
import numpy as np


class Conv1:

    def __init__(self, prop_count, values_count, patch_size, training):
        self.prop_count = prop_count
        self.values_count = values_count
        self.X = tf.placeholder(tf.float32, shape=(None, values_count, prop_count, patch_size))
        self.Y = tf.placeholder(tf.float32, shape=(None, prop_count, values_count))
        self.training = training
        self.cnn_model = None

    def model(self):

        # Conv 1
        model = tf.layers.conv2d(inputs=self.X, filters=30, padding='same', activation=tf.nn.relu,
                                 kernel_size=5, name='conv1')
        # pool 1
        model = tf.layers.max_pooling2d(inputs=model, pool_size=2, strides=2, name='max1')

        model = tf.layers.flatten(model)

        # dense
        model = tf.layers.dense(inputs=model, units=self.prop_count*self.values_count, activation=tf.nn.relu, name='fc1')

        self.cnn_model = model

        return model

    def loss(self):
        if self.cnn_model is None:
            raise Exception("Variable cnn_model is not initialised yet")

        return tf.losses.mean_squared_error(labels=tf.reshape(self.Y, [-1, self.prop_count*self.values_count]), predictions=self.cnn_model)

    def train(self):
        self.model()
        loss = self.loss()
        optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
        return optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    def fill_feed_dict(self, X, Y):
        return {self.X: np.swapaxes(X, 1, 3), self.Y: Y}

    def eval(self, x):
        pass
