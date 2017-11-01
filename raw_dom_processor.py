import os
from parser.reader import Reader

RAW_DIR = '/home/vedanshu/WebstormProjects/getter/data'

for folder, _, files in os.walk(RAW_DIR):
    for file in files:
        reader = Reader(folder, file)
        reader.process()
