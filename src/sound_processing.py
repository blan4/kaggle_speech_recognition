# -*- coding: utf-8 -*-

import numpy as np
from scipy.io import wavfile


def scale_sound_min_max(sound):
    sound = sound.copy().astype('float32')
    if np.max(sound) - np.min(sound) <= 0.0001:
        sound.fill(0.51)
        sound[::2] = 0.49
        return sound
    return (sound - np.min(sound)) / (np.max(sound) - np.min(sound))


def scale_sound_int(sound):
    sound = sound.astype(np.float32)
    sound = (sound + (np.iinfo(np.int16).max / 2)) / np.iinfo(np.int16).max
    return sound


def read_wav_file(fname):
    _, wav = wavfile.read(fname)
    try:
        return scale_sound_int(wav)
    except Exception as err:
        print("Warning: sound with empty data: {}. Cause: {}".format(fname, err))
        wav = wav.astype(np.float32)
        wav.fill(0.5)
        return wav
