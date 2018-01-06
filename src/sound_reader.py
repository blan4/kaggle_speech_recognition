# -*- coding: utf-8 -*-
from abc import abstractmethod

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
        return wav
