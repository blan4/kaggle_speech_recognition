# -*- coding: utf-8 -*-
import unittest

import numpy as np

from augmentation import SoundEffects


class TestUtils(unittest.TestCase):
    def test_fix_size(self):
        se = SoundEffects(5)
        ar = se._fix_size(np.array([1, 2]))
        print(ar)
        self.assertEqual(5, len(ar))

        ar = se._fix_size(np.array([1, 2, 3, 4, 5, 6]))
        print(ar)
        self.assertEqual(5, len(ar))

        ar = se._fix_size(np.array([1, 2, 3, 4, 5]))
        print(ar)
        self.assertEqual(5, len(ar))


if __name__ == '__main__':
    unittest.main()
