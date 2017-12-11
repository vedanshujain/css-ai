import unittest
from parser.encoder import Encoder


class TestEncoder(unittest.TestCase):

    def test_init(self):
        enc = Encoder()
        self.assertIsInstance(enc, Encoder)