import os
import json


class Reader:

    RAW_DIR = '/home/vedanshu/WebstormProjects/getter'
    OUTPUT_DIR = '/home/vedanshu/PycharmProjects/cssai-main/data'
    DICT_DIR = '/home/vedanshu/PycharmProjects/cssai-main/dict01.json'
    dict = {}
    tags = []
    style_names = []
    style_values = []

    def __init__(self):
        self.load_dict()
        self.page_data = {
            'map': {},
            'data': {}
        }
        pass

    def load_dict(self):
        with open(self.DICT_DIR) as dict_file:
            self.dict = json.load(dict_file)
            self.tags = self.dict['tags']
            self.style_names = self.dict['style_names']
            self.style_values = self.dict['style_values']

    def start_processing(self):
        for folder, _, files in os.walk(self.RAW_DIR):
            pass

    def process_file(self, file_path, file_name):
        with open(file_path) as data_file:
            json_data = json.load(data_file)
            node_id, page_data = self.process_data(json_data)


    def process_data(self, json_data, node_id = 0, page_data = {}):
        if json_data['tag'].lower() not in self.tags:
            return page_data
        children = []
        style_data = {}
        node_id = node_id + 1
        for style_name, style_value in json_data['styles'].items():
            style_index, style_value_index = self.compute_style_index(style_name, style_value)
            style_data[style_index] = style_value_index

        for bound_name, bound_value in json_data['bound'].items():
            style_name = "bound_{}".format(bound_name)
            style_index, style_value_index = self.compute_style_index(style_name, bound_value)
            style_data[style_index] = style_value_index

        if "map" not in page_data.keys():
            page_data['map'] = {}

        if "data" not in page_data.keys():
            page_data['data'] = {}

        page_data['data'][node_id] = style_data
        for child in json_data['children']:
            child_node_id, page_data = self.process_data(child, node_id, page_data)
            children.append(child_node_id)

        page_data['map'][node_id] = children
        return node_id, page_data

    def compute_style_index(self, style_name, style_value):
        if style_name not in self.style_names:
            self.style_names.append(style_name)
            self.style_values.append([])
        style_index = self.style_names.index(style_name)

        if style_value not in self.style_values[style_index]:
            self.style_values[style_index].append(style_value)
        style_value_index = self.style_values[style_index].index(style_value)
        return style_index, style_value_index
