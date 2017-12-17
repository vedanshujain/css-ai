from selenium import webdriver
from parser.dictionary import Dictionary
from models.conv1 import Conv1 as Model
from parser.encoder import Encoder
import random
import tensorflow as tf
import numpy as np

driver = webdriver.Chrome('store/chromedriver')

model = Model(Dictionary.CSS_PROP_COUNT, Dictionary.CSS_VALUES_COUNT, Dictionary.PATCH_SIZE)

eval_op = model.eval()
train_op = model.train()

sess = tf.Session()

print("Loading checkpoint")
saver = tf.train.Saver()
saver.restore(sess, 'checkpoint/conv1')

css_props_script = open('store/js/initialize_css_props.js', 'r').read()

html_parse_script = open('store/js/html_parser.js', 'r').read()

apply_style_script = open('store/js/apply_style.js', 'r').read()

encoder = Encoder()

def eval(iterations):
    run_id = random.getrandbits(16)
    print("Starting run with ID: {}".format(run_id))
    print("Fetching DOM of currently opened page")
    driver.get('https://news.ycombinator.com/')
    driver.execute_script(css_props_script)
    driver.execute_script(html_parse_script)
    driver.execute_script(apply_style_script)
    style_data = driver.execute_script("return buildTree('cssai-{}')".format(run_id))
    print("Encoding data for ML processing")
    encoded_data = encoder.encode_eval_generator(style_data)
    for i in range(iterations):
        print("Starting iteration {}".format(i))
        ele_ids = []
        X = []
        Y = []
        for patches in encoded_data:
            for element_id, patch in patches.items():
                Y.append(patch[0])
                X.append(patch[1:])
                ele_ids.append(element_id)
        # do a minor train op with 20 iterations?
        print("starting minor train")
        for i in range(5):
            sess.run(train_op, feed_dict=model.fill_feed_dict(X, Y))
            print("Done {}".format(i))
        print('Ended minor train')
        y_pred = sess.run(eval_op, feed_dict=model.fill_feed_dict(X, Y))
        y_pred_matrix = np.reshape(y_pred, (-1, Dictionary.CSS_PROP_COUNT, Dictionary.CSS_VALUES_COUNT))
        print("Applying changes")
        apply_pred(y_pred_matrix, Y, ele_ids, run_id, driver)
        return


def apply_pred(y_pred_matrix, orig_matrix, elements, run_id, driver):
    changed_elements = get_changed_elements(y_pred_matrix, orig_matrix, elements)
    changed_styles = encoder.deflate(changed_elements)
    style_text = get_style_text(changed_styles, run_id)
    driver.execute_script("window.applyStyles(`{}`)".format(style_text))


def get_style_text(styles, run_id):
    style_string = ""
    for element_id, changed_styles in styles.items():
        style_string += "\n .cssai-{}-{} {{\n".format(run_id, element_id)
        for style_values in changed_styles:
            if style_values[0] is None:
                continue
            style_string += "{}: {} ; \n".format(style_values[0], style_values[1])
        style_string += "}"
    return style_string


def get_changed_elements(y_pred_matrix, orig_matrix, elements):
    changed_elements = {}
    for index, new_style in enumerate(y_pred_matrix):
        orig_style = orig_matrix[index]
        element_id = elements[index]
        new_style = (new_style == np.max(new_style, 1, keepdims=True)).astype(int)
        changed_styles = {}
        for style_index in range(Dictionary.CSS_PROP_COUNT):
            if np.sum(np.abs(orig_style[style_index] - new_style[style_index])) > 0:
                changed_styles[style_index] = new_style[style_index]
        if len(changed_styles.keys()) > 0:
            changed_elements[element_id] = changed_styles
    return changed_elements

eval(1)

