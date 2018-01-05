# -*- coding: utf-8 -*-
import os
from glob import glob

import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.metrics import confusion_matrix

from consts import id2name, L
from data_loader import load_train_data, valid_generator, get_silence, test_generator
from sound_chain import SoundChain
from sound_processing import AdjustLenWavProcessor, EmphasisWavProcessor, ReshapeWavProcessor
from sound_reader import SimpleWavFileReader


def main_predict(params):
    test_paths = glob(params['test_path'])
    if params['sample']:
        print("Get small sample")
        test_paths = test_paths[:params['sample_size']]
    model = load_model(params['model_path'])
    train_data, validate_data = load_train_data(params['audio_path'], params['validation_list_path'])
    assert len(train_data) != 0
    assert len(validate_data) != 0

    wav_reader = SimpleWavFileReader()
    silence_data = get_silence(train_data, wav_reader)
    sound_chain = SoundChain(
        SimpleWavFileReader(),
        AdjustLenWavProcessor(silence_data, L, L),
        EmphasisWavProcessor(silence_data, L, L, 0.97),
        ReshapeWavProcessor(silence_data, L, L),
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
    with open(params['submission_path'], 'w') as fout:
        fout.write('fname,label\n')
        for fname, label in submission.items():
            fout.write('{},{}\n'.format(fname, label))
    print('SAVED')


def main_confusion_matrix(params):
    model = load_model(params['model_path'])
    train_df, valid_df = load_train_data(params['audio_path'], params['validation_list_path'])
    assert len(train_df) != 0
    assert len(valid_df) != 0

    wav_reader = SimpleWavFileReader()
    silence_data = get_silence(train_df, wav_reader)
    sound_chain = SoundChain(
        SimpleWavFileReader(),
        AdjustLenWavProcessor(silence_data, L, L),
        EmphasisWavProcessor(silence_data, L, L, 0.97),
        ReshapeWavProcessor(silence_data, L, L),
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
