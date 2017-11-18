# -*- coding: utf-8 -*-
import random
import re
from glob import glob

import numpy as np
import pandas as pd

from consts import L, LABELS, name2id
from sound_processing import read_wav_file
from utils import to_categorical


def load_train_data(audio_path, validation_list_path):
    """ Return 2 lists of tuples:
    [(class_id, user_id, path), ...] for train
    [(class_id, user_id, path), ...] for validation
    """
    # Just a simple regexp for paths with three groups:
    # prefix, label, user_id
    pattern = re.compile("(.+\/)?(\w+)\/([^_]+)_.+wav")
    all_files = glob(audio_path)

    with open(validation_list_path, 'r') as fin:
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

            sample = (label, label_id, uid, entry)
            if uid in valset:
                val.append(sample)
            else:
                train.append(sample)

    columns_list = ['label', 'label_id', 'user_id', 'wav_file']
    print('There are {} train and {} validate samples'.format(len(train), len(val)))

    train_df = pd.DataFrame(train, columns=columns_list)
    valid_df = pd.DataFrame(val, columns=columns_list)
    return train_df, valid_df


def get_silence(train_df):
    silence_files = train_df[train_df.label == 'silence']
    silence_data = np.concatenate([read_wav_file(x) for x in silence_files.wav_file.values])
    return silence_data


def process_wav_file(fname, silence_data):
    wav = read_wav_file(fname)

    if len(wav) > L:
        i = np.random.randint(0, len(wav) - L)
        wav = wav[i:(i + L)]
    elif len(wav) < L:
        rem_len = L - len(wav)
        i = np.random.randint(0, len(silence_data) - rem_len)
        silence_part = silence_data[i:(i + L)]
        j = np.random.randint(0, rem_len)
        silence_part_left = silence_part[0:j]
        silence_part_right = silence_part[j:rem_len]
        wav = np.concatenate([silence_part_left, wav, silence_part_right])

    return wav.reshape((L, 1))


def train_generator(train_df: pd.DataFrame, silence_data, batch_size, n=2000):
    train_df = train_df[train_df.label != 'silence']  # TODO: WHY????
    while True:
        this_train = train_df.groupby('label_id').apply(lambda x: x.sample(n=n))
        shuffled_ids = random.sample(range(this_train.shape[0]), this_train.shape[0])
        for start in range(0, len(shuffled_ids), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(shuffled_ids))
            i_train_batch = shuffled_ids[start:end]
            for i in i_train_batch:
                x_batch.append(process_wav_file(this_train.wav_file.values[i], silence_data))
                y_batch.append(this_train.label_id.values[i])
            x_batch = np.array(x_batch)
            y_batch = to_categorical(y_batch, num_classes=len(LABELS))
            yield x_batch, y_batch


def valid_generator(valid_df, silence_data, batch_size, with_y=True):
    while True:
        ids = list(range(valid_df.shape[0]))
        for start in range(0, len(ids), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(ids))
            i_val_batch = ids[start:end]
            for i in i_val_batch:
                x_batch.append(process_wav_file(valid_df.wav_file.values[i], silence_data))
                y_batch.append(valid_df.label_id.values[i])
            x_batch = np.array(x_batch)
            y_batch = to_categorical(y_batch, num_classes=len(LABELS))

            if with_y:
                yield x_batch, y_batch
            else:
                yield x_batch


def get_sample_data(train_df, valid_df, n=30):
    t = train_df[train_df.label != 'silence'] \
        .groupby('label_id').apply(lambda x: x.sample(n=n)) \
        .reset_index(drop=['label_id'])
    v = valid_df[valid_df.label != 'silence'] \
        .groupby('label_id').apply(lambda x: x.sample(n=n)) \
        .reset_index(drop=['label_id'])

    return t, v


def test_generator(test_paths, batch_size, silence_data):
    # test_paths = glob(audio_path)
    while True:
        for start in range(0, len(test_paths), batch_size):
            x_batch = []
            end = min(start + batch_size, len(test_paths))
            this_paths = test_paths[start:end]
            for x in this_paths:
                x_batch.append(process_wav_file(x, silence_data))
            x_batch = np.array(x_batch)
            yield x_batch
