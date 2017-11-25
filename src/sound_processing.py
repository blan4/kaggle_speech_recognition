# -*- coding: utf-8 -*-

import numpy as np
from scipy.io import wavfile

from consts import L, stride_size, frame_size, strides


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


def process_wav_file(fname, silence_data):
    wav = read_wav_file(fname)
    wav = adjust_len(wav, silence_data)
    return wav.reshape((L, 1))


def process_wav_file_to_2d(fname, silence_data):
    wav = read_wav_file(fname)
    wav = adjust_len(wav, silence_data)
    return windowed_sound(wav)


def adjust_len(wav, silence_data):
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
        i = np.random.randint(0, len(silence_data) - rem_len)
        silence_part = silence_data[i:(i + L)]
        j = np.random.randint(0, rem_len)
        silence_part_left = silence_part[0:j]
        silence_part_right = silence_part[j:rem_len]
        return np.concatenate([silence_part_left, wav, silence_part_right])
