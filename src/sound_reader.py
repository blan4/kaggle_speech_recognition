# -*- coding: utf-8 -*-
import numpy as np
from scipy.io import wavfile

from sound_processing import scale_sound_int


def read_wav_file(fname):
    _, wav = wavfile.read(fname)
    try:
        return scale_sound_int(wav)
    except Exception as err:
        print("Warning: sound with empty data: {}. Cause: {}".format(fname, err))
        wav = wav.astype(np.float32)
        wav.fill(0.5)
        return wav


def read_and_process(process):
    def _do(fname, silence):
        wav = read_wav_file(fname)
        return process(wav, silence)

    return _do
