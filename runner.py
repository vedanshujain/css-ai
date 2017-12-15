from models.conv1 import Conv1 as Model
from parser.encoder import Encoder
import tensorflow as tf


CSS_PROP_COUNT = 90
CSS_VALUES_COUNT = 50
PATCH_SIZE = 4

if __name__ == '__main__':
    model = Model(CSS_PROP_COUNT, CSS_VALUES_COUNT, PATCH_SIZE, True)

    encoder = Encoder()
    train_op = model.train()
    loss = model.loss()

    global_step_tensor = tf.Variable(10, trainable=False, name='global_step')

    saver = tf.train.Saver()

    i = 0
    count = 0
    print("Starting session")
    with tf.Session() as sess:
        print("Session started")
        patches_generater = encoder.get_next_sample(100)
        print("Running global variables initializer")
        # sess.run(tf.global_variables_initializer())
        saver.restore(sess, 'checkpoint/conv1')
        for patches in patches_generater:
            print("Processing patches")
            Y = [patch[0] for patch in patches]
            X = [patch[1:] for patch in patches]
            _, loss_value = sess.run([train_op, loss], feed_dict=model.fill_feed_dict(X, Y))
            i += len(patches)
            count += 1
            if count % 10 == 0:
                print('Checkpointing')
                saver.save(sess, 'checkpoint/conv1')
                print('Checkpointed')
            print("Done {} iterations. Loss value: {}.".format(i, loss_value))

