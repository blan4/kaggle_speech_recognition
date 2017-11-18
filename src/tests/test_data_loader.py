# -*- coding: utf-8 -*-
import unittest

import numpy as np
from sklearn.metrics import confusion_matrix

from data_loader import load_train_data, get_silence
import pandas as pd


class TestDataLoader(unittest.TestCase):
    def test_load_data(self):
        train_df, valid_df = load_train_data('../../data/train/audio/*/**.wav', '../../data/train/validation_list.txt')
        self.assertTrue(train_df.shape[0] == 57929)
        self.assertTrue(valid_df.shape[0] == 6798)
        # print(train_df.groupby('label').count())
        classes = np.full(shape=valid_df.shape[0], fill_value='unknown')
        labels = np.unique(valid_df['label'].values)
        df = pd.DataFrame(confusion_matrix(valid_df['label'].values, classes, labels), columns=labels, index=labels)
        print(df)

    def test_get_silence(self):
        train_df, valid_df = load_train_data('../../data/train/audio/*/**.wav', '../../data/train/validation_list.txt')
        get_silence(train_df)


if __name__ == '__main__':
    unittest.main()
