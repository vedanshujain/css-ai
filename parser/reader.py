import os
import json
import numpy as np


class Reader:

    OUTPUT_DIR = '/home/vedanshu/PycharmProjects/cssai-main/data'
    DICT_DIR = '/home/vedanshu/PycharmProjects/cssai-main/dict01.json'
    dict = {}
    tags = []
    style_names = []
    style_values = []
    input_path = ''
    input_file_name = ''
    last_node_id = 0

    def __init__(self, input_path='', input_file_name=''):
        self.load_dict()
        self.page_data = {
            'map': {},
            'data': {}
        }
        self.input_path = input_path
        self.input_file_name = input_file_name
        self.last_node_id = 0

    def load_dict(self):
        with open(self.DICT_DIR) as dict_file:
            self.dict = json.load(dict_file)
            self.tags = self.dict['tags']
            self.style_names = self.dict['style_names']
            self.style_values = self.dict['style_values']

    def process(self):
        file_path = os.path.join(self.input_path, self.input_file_name)
        print("Processing {}", file_path)
        self.process_file(file_path, self.input_file_name)
        self.update_dict()

    def update_dict(self):
        dict_data = {
            'style_names': self.style_names,
            'style_values': self.style_values,
            'tags': self.tags
        }
        with open(self.DICT_DIR, 'w') as dict_file:
            json.dump(dict_data, dict_file, indent=2)

    def process_file(self, file_path, file_name):
        with open(file_path) as data_file:
            json_data = json.load(data_file)
            _, page_data = self.process_data(json_data)
            with open("{}/{}".format(self.OUTPUT_DIR, file_name), 'w') as outfile:
                json.dump(page_data, outfile, indent=2)

    def process_data(self, json_data, page_data={}):
        if json_data['tag'].lower() not in self.tags:
            return None, None
        children = []
        node_id = self.last_node_id + 1
        self.last_node_id = node_id
        style_data = {}
        print("Processing node id: {}".format(node_id))
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
            child_node_id, child_page_data = self.process_data(child, page_data)
            if child_node_id is not None:
                children.append(child_node_id)
                page_data = child_page_data

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

    def get_next_patch(self, file_path, file_name, count):
        data = Reader.load_json_file(file_path, file_name)
        current_count = 0
        style_data = []
        for parent in data['map'].keys():
            for element in data['map'][parent]:
                patch = self.get_patch(element, parent, data)
                style_data.append([
                    data['data'][str(patch['ele'])],
                    data['data'][str(patch['parent'])],
                    data['data'][str(patch['left_sib'])] if patch['left_sib'] is not None else None,
                    data['data'][str(patch['right_sib'])] if patch['right_sib'] is not None else None,
                    data['data'][str(patch['f_child'])] if patch['f_child'] is not None else None
                ])
                current_count += 1
                if current_count >= count:
                    yield [style_data, False]
                    style_data = []
                    current_count = 0
        yield [style_data, True]

    # noinspection PyMethodMayBeStatic
    def get_patch(self, element, parent, data):
        siblings = data['map'][parent]
        c_index = siblings.index(int(element))
        left_sib = siblings[c_index - 1] if c_index > 0 else None
        right_sib = siblings[c_index + 1] if c_index + 1 < len(siblings) else None
        f_child = data['map'][str(element)][0] if len(data['map'][str(element)]) > 0 else None
        return {
            'ele': str(element) if element is not None else None,
            'parent': str(parent) if parent is not None else None,
            'left_sib': str(left_sib) if left_sib is not None else None,
            'right_sib': str(right_sib) if right_sib is not None else None,
            'f_child': str(f_child) if f_child is not None else None
        }

    # noinspection PyMethodMayBeStatic
    def inflate_patches(self, patches):
        inflated_patches = []
        for patch in patches:
            inflated_patch = []
            for element in patch:
                inflated_element = []
                if element is None:
                    inflated_element = np.zeros((90, 50)).tolist()
                else:
                    for style_index in element.keys():
                        style_value = int(element[style_index])
                        ele_style = np.zeros(50).tolist()
                        if style_value < 50:
                            ele_style[style_value] = 1
                        inflated_element.append(ele_style)
                inflated_patch.append(inflated_element)
            inflated_patches.append(inflated_patch)
        return inflated_patches

    @staticmethod
    def load_json_file(file_path, file_name):
        full_file_path = os.path.join(file_path, file_name)
        with open(full_file_path) as data_file:
            return json.load(data_file)
