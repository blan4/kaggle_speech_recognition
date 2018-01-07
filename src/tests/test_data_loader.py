# -*- coding: utf-8 -*-
import unittest

from consts import L, LABELS
from data_loader import load_train_data, sampling, train_generator
from sound_chain import SoundChain
from sound_reader import get_silence, SimpleWavFileReader
import sound_processing as sp


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.audio_path = '../../../data/train/audio/*/**.wav'
        self.validation_list_path = '../../../data/train/validation_list.txt'

    def test_load_data(self):
        train_df, valid_df = load_train_data(self.audio_path, self.validation_list_path)
        self.assertTrue(train_df.shape[0] == 57929)
        self.assertTrue(valid_df.shape[0] == 6798)
        df = train_df.groupby('label').apply(sampling(2000))
        print(df.shape)

    def test_train_generator(self):
        train_df, valid_df = load_train_data(self.audio_path, self.validation_list_path)
        wav_reader = SimpleWavFileReader(L)
        silence_data = get_silence(train_df, wav_reader)
        train_sound_chain = SoundChain(
            SimpleWavFileReader(L),
            sp.AdjustLenWavProcessor(silence_data, L, L),
            sp.EmphasisWavProcessor(silence_data, L, L, 0.97),
            sp.NormalizeWavProcessor(silence_data, L, L),
            sp.ReshapeWavProcessor(silence_data, L, L),
            sp.MinMaxWavProcessor(silence_data, L, L, (0, 1)),
        )
        n = 2
        gen = train_generator(train_df, 64, train_sound_chain, n)
        batch = gen.__next__()
        self.assertEqual(batch[0].shape, (len(LABELS) * n, L, 1))
        self.assertEqual(batch[1].shape, (len(LABELS) * n, len(LABELS)))


if __name__ == '__main__':
    unittest.main()
