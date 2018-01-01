from parser.encoder import Encoder
from parser.reader import Reader
import tensorflow as tf
import numpy as np


class Conv1:

    def __init__(self, prop_count, values_count, patch_size, style_index):
        self.prop_count = prop_count
        self.values_count = values_count
        self.X = tf.placeholder(tf.float32, shape=(None, prop_count, values_count, patch_size))
        self.Y = tf.placeholder(tf.float32, shape=(None, prop_count, values_count))
        self.cnn_model = None
        self.style_index = style_index

    def model(self):

        if self.cnn_model is not None:
            return

        # Conv 1
        conv1 = tf.layers.conv2d(inputs=self.X, filters=1024, padding='valid', activation=tf.nn.tanh,
                                 kernel_size=(1, 150), name='conv1')

        tf.summary.histogram('layer-conv1', conv1)

        # Conv 2
        conv2 = tf.layers.conv2d(inputs=conv1, strides=1, filters=512, padding='same', activation=tf.nn.tanh,
                                 kernel_size=(1, 75), name='conv2')

        tf.summary.histogram('layer-conv2', conv2)

        # Conv 3
        conv3 = tf.layers.conv2d(inputs=conv2, filters=256, padding='same', activation=tf.nn.tanh,
                                 kernel_size=(1, 33), name='conv3')

        tf.summary.histogram('layer-conv4', conv3)

        # pool 1
        # pool1 = tf.layers.max_pooling2d(inputs=conv3, pool_size=1, strides=(1, 2), name='max1', padding='same')

        # tf.summary.histogram('layer-pool1', pool1)

        # Conv 4
        # merge all css props
        conv4 = tf.layers.conv2d(inputs=conv3, strides=1, filters=128, padding='same', activation=tf.nn.tanh,
                                 kernel_size=(1, 15), name='conv4')

        tf.summary.histogram('layer-conv4', conv4)

        # pool 2
        # pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=1, strides=1, name='max2')

        model = tf.layers.flatten(conv4)

        # dense
        fc1 = tf.layers.dense(inputs=model, units=self.values_count * self.prop_count, activation=tf.nn.tanh, name='fc1')
        tf.summary.histogram('layer-fc1', fc1)
        fc2 = tf.layers.dense(inputs=fc1, units=self.values_count, activation=tf.nn.tanh, name='fc2')
        tf.summary.histogram('layer-fc2', fc2)

        # normalizing probabilities
        model = tf.nn.softmax(fc2)

        self.cnn_model = model

        return model

    def loss(self):
        if self.cnn_model is None:
            raise Exception("Variable cnn_model is not initialised yet")

        input_count = tf.cast(tf.shape(self.X)[0], tf.float32)

        if self.style_index is not None:
            y_style = tf.gather(self.Y, axis=1, indices=self.style_index)
            pred_style = self.cnn_model
            loss_op = tf.nn.softmax_cross_entropy_with_logits(labels=y_style, logits=pred_style)
            argmax_y_op = tf.argmax(y_style, axis=-1)
            argmax_pred_op = tf.argmax(pred_style, axis=-1)
            tf.summary.histogram('y_style', argmax_y_op)
            tf.summary.histogram('pred_style', argmax_pred_op)
            return tf.reduce_sum(tf.Print(loss_op,
                                 [tf.not_equal(argmax_pred_op, argmax_y_op),
                                  argmax_y_op, argmax_pred_op,
                                  tf.reduce_sum(pred_style), tf.reduce_sum(y_style)], summarize=50)) / input_count

        return tf.reduce_sum(
           tf.square(tf.subtract(self.Y, self.cnn_model))
        ) / (input_count * 2)

    def train(self):
        self.model()
        loss = self.loss()
        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.1)
        return optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    def fill_feed_dict(self, X, Y):
        return {self.X: np.swapaxes(np.swapaxes(X, 1, 3), 1, 2), self.Y: Y}

    def eval(self):
        return self.model()

