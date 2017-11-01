import os
import json


class Reader:

    RAW_DIR = '/home/vedanshu/WebstormProjects/getter'
    OUTPUT_DIR = '/home/vedanshu/PycharmProjects/cssai-main/data'
    DICT_DIR = '/home/vedanshu/PycharmProjects/cssai-main/dict.dat'
    dict = {}

    def __init__(self):
        self.load_dict()
        pass

    def load_dict(self):
        with open(self.DICT_DIR) as dict_file:
            self.dict = json.load(dict_file)

    def start_processing(self):
        for _, _, files in os.walk(self.RAW_DIR):
            pass

