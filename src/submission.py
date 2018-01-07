# -*- coding: utf-8 -*-
import os
from datetime import datetime
from glob import glob

import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.metrics import confusion_matrix

import sound_processing as sp
from consts import id2name, L
from data_loader import load_train_data, valid_generator, test_generator
from sound_chain import SoundChain
from sound_reader import SimpleWavFileReader, get_silence


def main_predict(params):
    submission_path = os.path.join(params['submission_path'], str(datetime.now()))
    os.makedirs(submission_path, exist_ok=True)

    with open(os.path.join(submission_path, "submission.csv"), 'w') as fout:
        submission = _make_submission(params)

        fout.write('fname,label\n')
        for fname, label in submission.items():
            fout.write('{},{}\n'.format(fname, label))
    print('SAVED')


def _make_submission(params):
    test_paths = glob(params['test_path'])
    if params['sample']:
        print("Get small sample")
        test_paths = test_paths[:params['sample_size']]
    model = load_model(params['model_path'])
    train_data, validate_data = load_train_data(params['audio_path'], params['validation_list_path'])
    assert len(train_data) != 0
    assert len(validate_data) != 0

    wav_reader = SimpleWavFileReader(L)
    silence_data = get_silence(train_data, wav_reader)
    sound_chain = SoundChain(
        SimpleWavFileReader(L),
        sp.AdjustLenWavProcessor(silence_data, L, L),
        sp.EmphasisWavProcessor(silence_data, L, L, 0.97),
        sp.NormalizeWavProcessor(silence_data, L, L),
        sp.ReshapeWavProcessor(silence_data, L, L),
        sp.MinMaxWavProcessor(silence_data, L, L, (0, 1)),
    )

    tests = test_generator(test_paths, params['batch_size_pred'], sound_chain)
    print("PREDICTING")
    predictions = model.predict_generator(tests, int(np.ceil(len(test_paths) / params['batch_size_pred'])))
    classes = np.argmax(predictions, axis=1)
    submission = {}
    print("SAVING")
    for i in range(len(test_paths)):
        fname, label = os.path.basename(test_paths[i]), id2name[classes[i]]
        submission[fname] = label

    return submission


def main_confusion_matrix(params):
    model = load_model(params['model_path'])
    train_df, valid_df = load_train_data(params['audio_path'], params['validation_list_path'])
    assert len(train_df) != 0
    assert len(valid_df) != 0

    wav_reader = SimpleWavFileReader(L)
    silence_data = get_silence(train_df, wav_reader)
    sound_chain = SoundChain(
        SimpleWavFileReader(L),
        sp.AdjustLenWavProcessor(silence_data, L, L),
        sp.EmphasisWavProcessor(silence_data, L, L, 0.97),
        sp.NormalizeWavProcessor(silence_data, L, L),
        sp.ReshapeWavProcessor(silence_data, L, L),
        sp.MinMaxWavProcessor(silence_data, L, L, (0, 1)),
    )

    validate_gen = valid_generator(valid_df, params['batch_size'], sound_chain, with_y=False)
    predictions = model.predict_generator(validate_gen, int(np.ceil(valid_df.shape[0] / params['batch_size_pred'])))
    classes = [id2name[i] for i in np.argmax(predictions, axis=1)]
    y_true = valid_df['label'].values
    labels = np.unique(valid_df['label'].values)
    cm = confusion_matrix(y_true, classes, labels=labels)
    df = pd.DataFrame(cm, columns=labels, index=labels)
    df.to_csv(os.path.join(params['output_path'], 'confusion.csv'), index_label='index')
    print(df)
    return df
