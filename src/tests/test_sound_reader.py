# -*- coding: utf-8 -*-
import unittest
from glob import glob

from data_loader import load_train_data
from sound_reader import read_wav_file


class TestSoundReader(unittest.TestCase):
    """
    This test just check files and the algorithm of reading it.
    """

    def test_read_wav_train(self):
        train_df, valid_df = load_train_data('../../data/train/audio/*/**.wav', '../../data/train/validation_list.txt')
        self.assertTrue(train_df.shape[0] == 57929)
        self.assertTrue(valid_df.shape[0] == 6798)
        for _, file in train_df['wav_file'].iteritems():
            w = read_wav_file(file)
            self.assertLessEqual(w.max(), 1.0, file)
            self.assertGreaterEqual(w.max(), 0.0, file)
        for _, file in valid_df['wav_file'].iteritems():
            w = read_wav_file(file)
            self.assertLessEqual(w.max(), 1.0, file)
            self.assertGreaterEqual(w.max(), 0.0, file)

    def test_read_wav_test(self):
        files = glob("../../data/test/audio/*.wav")
        print("{} test samples".format(len(files)))
        for file in files:
            w = read_wav_file(file)
            self.assertLessEqual(w.max(), 1.0, file)
            self.assertGreaterEqual(w.max(), 0.0, file)


if __name__ == '__main__':
    unittest.main()
