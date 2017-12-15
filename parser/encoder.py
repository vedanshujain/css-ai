from .reader import Reader
import os


class Encoder:

    reader = None
    output_dir = None
    net_count = 0

    def __init__(self):
        self.reader = Reader()
        self.reader.load_dict()

    def get_next_sample(self, try_count=10):
        for folder, _, files in os.walk(self.reader.OUTPUT_DIR):
            for file in files:
                file_parsed = False
                while not file_parsed:
                    for results in self.reader.get_next_patch(folder, file, try_count):
                        patches, file_parsed = results
                        inflated_patches = self.reader.inflate_patches(patches)
                        yield inflated_patches

    def encode_eval_generator(self, style_data, count=100):
        _, processed_data = self.reader.process_data(style_data, read_only=True)
        style_data = self.reader.get_style_data(processed_data, count)
        for patches, _ in style_data:
            inflated_patches = self.reader.inflate_patches(patches)
            yield inflated_patches
