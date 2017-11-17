# -*- coding: utf-8 -*-
import unittest

from utils import to_categorical


class TestUtils(unittest.TestCase):
    def test_to_categorical(self):
        y = to_categorical(1, 3)
        self.assertEquals(y.shape, (1, 3))
        self.assertListEqual(list(y[0]), [0, 1, 0])


if __name__ == '__main__':
    unittest.main()
