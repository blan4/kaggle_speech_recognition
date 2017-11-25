# -*- coding: utf-8 -*-

import numpy as np
from scipy.io import wavfile

sr = 16000
ms_to_s = 1000.0
frame_size = int((20 / ms_to_s) * sr)
stride_size = int((10 / ms_to_s) * sr)
strides = int(sr / stride_size)


def scale_sound_min_max(sound):
    sound = sound.copy().astype('float32')
    if np.max(sound) - np.min(sound) <= 0.0001:
        sound.fill(0.51)
        sound[::2] = 0.49
        return sound
    return (sound - np.min(sound)) / (np.max(sound) - np.min(sound))


def scale_sound_int(sound):
    m = np.iinfo(np.int16).max
    sound = (sound.astype('int32') + m) / (2.0 * m)
    return sound.astype('float32')


def read_wav_file(fname):
    _, wav = wavfile.read(fname)
    try:
        return scale_sound_int(wav)
    except Exception as err:
        print("Warning: sound with empty data: {}. Cause: {}".format(fname, err))
        wav = wav.astype(np.float32)
        wav.fill(0.5)
        return wav


def windowed_sound(wav):
    w = np.pad(wav, int(stride_size / 2), mode='constant', constant_values=0.5)
    ww = np.vstack([w[i * stride_size:i * stride_size + frame_size].reshape(1, frame_size) for i in range(strides)])
    return ww
