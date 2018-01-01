from .reader import Reader
import os
import random
import numpy as np

class Encoder:

    reader = None
    output_dir = None
    net_count = 0

    def __init__(self):
        self.reader = Reader()
        self.reader.load_dict()

    def get_next_sample(self, try_count=10, balance_index = None):
        for folder, _, files in os.walk(self.reader.OUTPUT_DIR):
            random.shuffle(files)
            for file in files:
                print("Processing file: {}", file)
                file_parsed = False
                while not file_parsed:
                    for results in self.reader.get_next_patch(folder, file, try_count):
                        patches, file_parsed = results
                        inflated_patches = self.reader.inflate_patches(patches)
                        if balance_index is not None:
                            style_data = []
                            element_ids = []
                            for element_id, patch in inflated_patches.items():
                                style_data.append(patch[0][balance_index])
                                element_ids.append(element_id)
                            res, indexes = np.unique(style_data, axis=0, return_index=True)
                            ele_ids = [element_ids[i] for i in indexes]
                            filtered_inflated_patches = {}
                            for id in ele_ids:
                                filtered_inflated_patches[id] = inflated_patches[id]
                            inflated_patches = filtered_inflated_patches
                        yield inflated_patches

    def encode_eval_generator(self, style_data, count=100):
        _, processed_data = self.reader.process_data(style_data, read_only=True)
        style_data = self.reader.get_style_data(processed_data, count)
        for patches, _ in style_data:
            inflated_patches = self.reader.inflate_patches(patches)
            yield inflated_patches

    def deflate(self, changed_elements):
        deflated_styles = {}
        for element_id, styles in changed_elements.items():
            deflated_styles[element_id] = [self.reader.get_style_value(style_index, list(style_row).index(1))
                                           for style_index, style_row in styles.items() if 1 in style_row]
        return deflated_styles
