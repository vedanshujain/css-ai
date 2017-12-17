from models.conv1 import Conv1 as Model
from parser.encoder import Encoder
import tensorflow as tf
from parser.dictionary import Dictionary
import time
import os
import numpy as np


if __name__ == '__main__':
    model = Model(Dictionary.CSS_PROP_COUNT, Dictionary.CSS_VALUES_COUNT, Dictionary.PATCH_SIZE)

    encoder = Encoder()
    train_op = model.train()
    loss = model.loss()

    global_step_tensor = tf.Variable(10, trainable=False, name='global_step')

    tf.summary.scalar('entropy loss', loss)
    summary = tf.summary.merge_all()

    saver = tf.train.Saver()

    i = 0
    count = 0
    print("Starting session")
    with tf.Session() as sess:
        dir_path = "train/{}".format(int(time.time()))
        os.mkdir(dir_path)
        writer = tf.summary.FileWriter(dir_path, sess.graph)
        print("Session started")
        patches_generater = encoder.get_next_sample(100)
        print("Running global variables initializer")
        # sess.run(tf.global_variables_initializer())
        saver.restore(sess, 'checkpoint/conv1')
        for patches in patches_generater:
            print("Processing patches")
            Y = [patch[0] for _, patch in patches.items()]
            X = [patch[1:] for _, patch in patches.items()]
            _, loss_value, summary_str = sess.run([train_op, loss, summary], feed_dict=model.fill_feed_dict(X, Y))
            writer.add_summary(summary_str, count)
            i += len(patches)
            count += 1
            if count % 10 == 0:
                print('Checkpointing')
                saver.save(sess, 'checkpoint/conv1')
                print('Checkpointed')
            print("Done {} iterations. Loss value: {}.".format(i, loss_value))

