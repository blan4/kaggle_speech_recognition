# -*- coding: utf-8 -*-
import os
import re
from glob import glob

import numpy as np
import soundfile

from consts import L, LABELS, name2id, audio_path, validation_list_path
from utils import to_categorical


def load_train_data(data_dir):
    """ Return 2 lists of tuples:
    [(class_id, user_id, path), ...] for train
    [(class_id, user_id, path), ...] for validation
    """
    # Just a simple regexp for paths with three groups:
    # prefix, label, user_id
    pattern = re.compile("(.+\/)?(\w+)\/([^_]+)_.+wav")
    all_files = glob(os.path.join(data_dir, audio_path))

    with open(os.path.join(data_dir, validation_list_path), 'r') as fin:
        validation_files = fin.readlines()
    valset = set()
    for entry in validation_files:
        r = re.match(pattern, entry)
        if r:
            valset.add(r.group(3))

    train, val = [], []
    for entry in all_files:
        r = re.match(pattern, entry)
        if r:
            label, uid = r.group(2), r.group(3)
            if label == '_background_noise_':
                label = 'silence'
            if label not in LABELS:
                label = 'unknown'

            label_id = name2id[label]

            sample = (label_id, uid, entry)
            if uid in valset:
                val.append(sample)
            else:
                train.append(sample)

    print('There are {} train and {} validate samples'.format(len(train), len(val)))
    return train, val


def scale_sound(sound):
    """
    Min Max scaling
    :param sound:
    :return:
    """
    if np.max(sound) - np.min(sound) <= 0:
        raise ZeroDivisionError("Bad sound file: len={}".format(len(sound)))

    return ((sound - np.min(sound)) / (np.max(sound) - np.min(sound))).astype(np.float32)


def data_generator(data, batch_size, shuffle=True):
    if shuffle:
        np.random.shuffle(data)

    def generator():
        for (label_id, uid, fname) in data:
            try:
                wav, sr = soundfile.read(fname, dtype='int32')
                if sr != L or len(wav) < L:
                    continue
                wav = scale_sound(wav.astype(np.int64))
                beg = 0
                yield np.array([wav[beg: beg + L].reshape(L, 1)]), to_categorical(label_id, len(LABELS))
                """
                # let's generate more silence!
                samples_per_file = 1 if label_id != name2id['silence'] else 20
                for _ in range(samples_per_file):
                    if len(wav) > L:
                        beg = np.random.randint(0, len(wav) - L)
                    else:
                        beg = 0
                    yield np.array([wav[beg: beg + L].reshape(L, 1)]), to_categorical(label_id, len(LABELS))
                """
            except Exception as err:
                print(err, label_id, uid, fname)

    def batch_generator():
        X = np.empty((batch_size, L, 1))
        Y = np.empty((batch_size, len(LABELS)))
        i = 0
        for x, y in generator():
            X[i] = x
            Y[i] = y
            i += 1
            if i >= batch_size - 1:
                yield X, Y
                i = 0
                X = np.empty((batch_size, L, 1))
                Y = np.empty((batch_size, len(LABELS)))

    return batch_generator
