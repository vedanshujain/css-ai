import unittest
from parser.reader import Reader


class TestReader(unittest.TestCase):

    def test_get_patch(self):
        data = {
            'map': {
                '1': [2, 3, 4],
                '3': [5, 6]
            }
        }
        reader = Reader()
        patch = reader.get_patch('3', '1', data)
        self.assertDictEqual(patch, {
            'ele': '3',
            'parent': '1',
            'left_sib': '2',
            'right_sib': '4',
            'f_child': '5'
        })

    def test_get_next_patch(self):
        reader = Reader()
        path = "/home/vedanshu/PycharmProjects/cssai-main/tests"
        for data in reader.get_next_patch(path, "test_data.json", 10):
            self.assertEqual(data[1], True)
            self.assertEqual(len(data[0]), 4)

    def test_inflate_patches(self):
        reader = Reader()
        path = "/home/vedanshu/PycharmProjects/cssai-main/tests"
        patch = None
        for data in reader.get_next_patch(path, "test_data.json", 10):
            patch = data[0]
        inflated_patches = reader.inflate_patches(patch)
        self.assertEqual(inflated_patches[0][1][0][3], 0)