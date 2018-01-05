# -*- coding: utf-8 -*-
from abc import abstractmethod

import numpy as np

from augmentation import SoundEffects


def _adjust_len(wav, silence_data, L):
    """
    Some files are short, so we add some silence to adjust their size.
    :param wav:
    :param silence_data: used as filler
    :return: np.ndarray of the sound
    """
    if len(wav) > L:
        i = np.random.randint(0, len(wav) - L)
        return wav[i:(i + L)]
    elif len(wav) < L:
        rem_len = L - len(wav)
        i = np.random.randint(0, len(silence_data) - L)
        j = np.random.randint(0, rem_len)

        # add zeros to the left and right sides of the short audio
        # this zeros will be filled with noise
        new_wav = np.concatenate([np.zeros(j), wav, np.zeros(rem_len - j)])

        silence_part = silence_data[i:(i + L)]  # select random part of noise
        assert len(silence_part) == L
        assert len(new_wav) == L
        # add low volume noise to the entire chunk of data.
        # we cannot add noise to the left and right sides only it would led to over fitting
        return silence_part * 0.3 + new_wav

    return wav


def _emphasis(signal, alpha=0.97):
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])


def _scale_sound_int(sound):
    m = np.iinfo(np.int16).max
    sound = (sound.astype('int32') + m) / (2.0 * m)
    return sound.astype('float32')


def _scale_sound_min_max(sound):
    sound = sound.copy().astype('float32')
    if np.max(sound) - np.min(sound) <= 0.0001:
        sound.fill(0.51)
        sound[::2] = 0.49
        return sound
    return (sound - np.min(sound)) / (np.max(sound) - np.min(sound))


class WavProcessor:
    def __init__(self, silence_data, sample_rate, L) -> None:
        self._silence_data = silence_data
        self.sample_rate = sample_rate
        self.L = L

    @abstractmethod
    def process(self, wav):
        pass


class AdjustLenWavProcessor(WavProcessor):
    def process(self, wav):
        wav = _adjust_len(wav, self._silence_data, self.L)
        return wav


class EffectsWavProcessor(WavProcessor):
    def __init__(self, silence_data, sample_rate, L) -> None:
        super().__init__(silence_data, sample_rate, L)
        self.se = SoundEffects(sample_rate=sample_rate)

    def process(self, wav):
        wav = _adjust_len(wav, self._silence_data, self.L)
        wav = self.se.call(wav)
        return wav


class ReshapeWavProcessor(WavProcessor):
    def process(self, wav):
        return wav.reshape((self.L, 1))


class EmphasisWavProcessor(WavProcessor):
    def __init__(self, silence_data, sample_rate, L, alpha=0.97):
        super().__init__(silence_data, sample_rate, L)
        self.alpha = alpha

    def process(self, wav):
        return _emphasis(wav, alpha=self.alpha)


class NormalizeWavProcessor(WavProcessor):
    def process(self, wav):
        try:
            return _scale_sound_int(wav)
        except Exception as err:
            print("Warning: sound with empty data. Cause: {}".format(err))
            wav = wav.astype(np.float32)
            wav.fill(0.5)
            return wav


class AddNoiseWavProcessor(WavProcessor):
    def process(self, wav):
        i = np.random.randint(0, len(self._silence_data) - self.L)
        silence_part = self._silence_data[i:(i + self.L)]
        alpha = np.random.randint(0, 40) / 100.0
        return silence_part * alpha + wav
