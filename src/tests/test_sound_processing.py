# -*- coding: utf-8 -*-
import unittest

from sound_processing import process_wav_file_to_2d
from sound_reader import read_wav_file


class TestSoundProcessing(unittest.TestCase):
    def test_process_wav_file_to_2d(self):
        wav = read_wav_file("../../data/train/audio/yes/9c59dd28_nohash_0.wav")
        self.assertEquals(wav.shape[0], 16000)
        res = process_wav_file_to_2d(wav, [])


if __name__ == '__main__':
    unittest.main()
