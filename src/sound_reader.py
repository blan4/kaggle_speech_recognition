# -*- coding: utf-8 -*-
from abc import abstractmethod

import numpy as np
from librosa.core import load


class WavFileReader:
    @abstractmethod
    def read(self, fname):
        pass


class SimpleWavFileReader(WavFileReader):
    def __init__(self, sample_rate) -> None:
        super().__init__()
        self.sample_rate = sample_rate

    def read(self, fname):
        wav, _ = load(fname, sr=self.sample_rate)
        wav = np.nan_to_num(wav)
        return wav


def get_silence(train_df, reader: WavFileReader):
    silence_files = train_df[train_df.label == 'silence']
    silence_data = np.concatenate([reader.read(x) for x in silence_files.wav_file.values])
    return silence_data
