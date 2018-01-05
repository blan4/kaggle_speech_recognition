# -*- coding: utf-8 -*-
from abc import abstractmethod

from scipy.io import wavfile


class WavFileReader:
    @abstractmethod
    def read(self, fname):
        pass


class SimpleWavFileReader(WavFileReader):
    def read(self, fname):
        _, wav = wavfile.read(fname)
        return wav
