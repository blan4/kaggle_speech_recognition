# -*- coding: utf-8 -*-
import unittest

import numpy as np

from consts import L
from data_loader import load_train_data
from sound_reader import SimpleWavFileReader
from tests import config


class TestSoundReader(unittest.TestCase):
    """
    This test just check files and the algorithm of reading it.
    """

    def test_read_wav_train(self):
        sr = SimpleWavFileReader(L)
        train_df, valid_df = load_train_data(config.audio_path, config.validation_list_path)
        self.assertTrue(train_df.shape[0] == 57929)
        self.assertTrue(valid_df.shape[0] == 6798)
        for _, file in train_df['wav_file'].iteritems():
            w = sr.read(file)
            self.assertFalse(np.isnan(w).any())
        for _, file in valid_df['wav_file'].iteritems():
            w = sr.read(file)
            self.assertFalse(np.isnan(w).any())


if __name__ == '__main__':
    unittest.main()
