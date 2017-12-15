from selenium import webdriver
from parser.dictionary import Dictionary
from models.conv1 import Conv1 as Model
from parser.encoder import Encoder
import random
import tensorflow as tf

driver = webdriver.Chrome('store/chromedriver')

model = Model(Dictionary.CSS_PROP_COUNT, Dictionary.CSS_VALUES_COUNT, Dictionary.PATCH_SIZE)

eval_op = model.eval()

sess = tf.Session()

print("Loading checkpoint")
saver = tf.train.Saver()
saver.restore(sess, 'checkpoint/conv1')

css_props_script = open('store/js/initialize_css_props.js', 'r').read()

html_parse_script = open('store/js/html_parser.js', 'r').read()

encoder = Encoder()

def eval(iterations):
    run_id = random.getrandbits(16)
    print("Starting run with ID: {}".format(run_id))
    print("Fetching DOM of currently opened page")
    driver.execute_script(css_props_script)
    driver.execute_script(html_parse_script)
    style_data = driver.execute_script("return buildTree('cssai-{}')".format(run_id))
    print("Encoding data for ML processing")
    encoded_data = encoder.encode_eval_generator(style_data)
    for i in range(iterations):
        print("Starting iteration {}".format(i))
        for patches in encoded_data:
            Y = [patch[0] for patch in patches]
            X = [patch[1:] for patch in patches]
            y_pred = sess.run(eval_op, feed_dict=model.fill_feed_dict(X, Y))
            return

eval(1)

