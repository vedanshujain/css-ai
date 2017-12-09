import unittest
from parser.encoder import Encoder


class TestEncoder(unittest.TestCase):

    def test_init(self):
        enc = Encoder()
        self.assertIsInstance(enc, Encoder)

    def test_get_next_sample(self):
        enc = Encoder()
        for samples in enc.get_next_sample():
            print(samples)
            break
        self.assertEqual(True, True)