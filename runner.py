from models.conv1 import Conv1 as Model
from parser.encoder import Encoder
import tensorflow as tf
from parser.dictionary import Dictionary
import time
import os
import numpy as np


if __name__ == '__main__':
    model = Model(Dictionary.CSS_PROP_COUNT, Dictionary.CSS_VALUES_COUNT, Dictionary.PATCH_SIZE, 0)

    encoder = Encoder()
    train_op = model.train()
    loss = model.loss()

    global_step_tensor = tf.Variable(10, trainable=False, name='global_step')

    loss_summary = tf.summary.scalar('entropy loss', loss)
    # prediction_arg_max = tf.argmax(model.cnn_model, axis=-1)
    # output_arg_max = tf.argmax(model.Y, axis=-1)
    # for i in range(Dictionary.CSS_PROP_COUNT):
    #     summary_op = tf.gather(prediction_arg_max, indices=i, axis=-1)
    #     output_summary_op = tf.gather(output_arg_max, indices=i, axis=-1)
    #     tf.summary.histogram("{}-{}-prediction".format(i, encoder.reader.style_names[i]),
    #                          summary_op)
    #     tf.summary.histogram("{}-{}-output".format(i, encoder.reader.style_names[i]),
    #                          output_summary_op)

    summary = tf.summary.merge_all()

    saver = tf.train.Saver()

    i = 0
    count = 0
    print("Starting session")
    with tf.Session() as sess:
        loss_summary_path = "train/{}".format('7-learning-rate-0.1')
        os.mkdir(loss_summary_path)
        writer = tf.summary.FileWriter(loss_summary_path, sess.graph)
        print("Session started")
        patches_generator = encoder.get_next_sample(1500, balance_index=0)
        print("Running global variables initializer")
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, 'checkpoint/conv1')
        for patches in patches_generator:
            print("Processing patches")
            Y = [patch[0] for _, patch in patches.items()]
            X = [patch[1:] for _, patch in patches.items()]
            train_out, loss_value, summary_str = sess.run([train_op, loss, summary], feed_dict=model.fill_feed_dict(X, Y))
            writer.add_summary(summary_str, count)
            if loss_value <= 0.05:
                print('here')
            i += len(patches)
            count += 1
            if count % 100 == 0:
                print('Checkpointing')
                saver.save(sess, 'checkpoint/conv1')
                print('Checkpointed')
            print("Done {} iterations. Loss value: {}.".format(i, loss_value))

