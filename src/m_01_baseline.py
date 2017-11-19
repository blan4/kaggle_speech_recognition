# -*- coding: utf-8 -*-
import os
from glob import glob

import numpy as np
import pandas as pd
from keras import Input, metrics
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.engine import Model
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, K
from keras.models import load_model
from keras.optimizers import SGD
from sklearn.metrics import confusion_matrix

from consts import L, LABELS, id2name
from data_loader import load_train_data, train_generator, valid_generator, get_silence, get_sample_data, test_generator

NAME = "BaselineSpeech"


def build():
    inputs = Input(shape=(L, 1))  # 16000
    x = inputs
    conf = [
        ['c', 1, 5, 128],  # 16000x128
        ['p', 5],  # 3200x128
        ['c', 1, 5, 128],  # 3200x128
        ['p', 5],  # 640x128
        ['c', 1, 5, 256],  # 640x256
        ['p', 5],  # 128x256
        ['c', 1, 4, 256],  # 128x256
        ['p', 4],  # 32x256
        ['c', 1, 4, 512],  # 32x512
        ['p', 4],  # 8x512
        ['c', 1, 4, 512],  # 8x512
        ['p', 4],  # 2x512
        ['c', 1, 2, 512],  # 2x512
        ['p', 2],  # 1x512
        ['c', 1, 2, 512]  # 1x512
    ]

    for layer in conf:
        if layer[0] == 'c':
            x = Conv1D(filters=layer[3], kernel_size=layer[2], strides=layer[1])(x)
        elif layer[0] == 'p':
            x = MaxPooling1D(pool_size=layer[1])(x)
        else:
            print("Unknown layer")

    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(len(LABELS), activation='sigmoid')(x)  # 12

    return Model(inputs, x, name=NAME)


def train(model, train_gen, validation_gen, params):
    print(params)
    model.summary()
    model.compile(optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                  loss=K.categorical_crossentropy,
                  metrics=[metrics.categorical_accuracy])

    return model.fit_generator(
        generator=train_gen,
        steps_per_epoch=params['steps_per_epoch'],
        epochs=params['epochs'],
        validation_data=validation_gen,
        validation_steps=params['validation_steps'],
        callbacks=[TensorBoard(
            log_dir=params['tensorboard_dir'],
            write_images=True,
            batch_size=params['batch_size']
        ), ModelCheckpoint(
            os.path.join(params['chekpoints_path'],
                         "weights-improvement-{epoch:02d}-{val_categorical_accuracy:.2f}.hdf5"),
            monitor='val_categorical_accuracy',
            verbose=1,
            save_best_only=True,
            mode='auto'
        )])


def main_predict(params):
    test_paths = glob(params['test_path'])
    if params['sample']:
        print("Get small sample")
        test_paths = test_paths[:params['sample_size']]
    model = load_model(params['model_path'])
    train_data, validate_data = load_train_data(params['audio_path'], params['validation_list_path'])
    assert len(train_data) != 0
    assert len(validate_data) != 0
    silence_data = get_silence(train_data)
    tests = test_generator(test_paths, params['batch_size_pred'], silence_data)
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
    silence_data = get_silence(train_df)
    validate_gen = valid_generator(valid_df, silence_data, params['batch_size'], with_y=False)
    predictions = model.predict_generator(validate_gen, int(np.ceil(valid_df.shape[0] / params['batch_size_pred'])))
    classes = [id2name[i] for i in np.argmax(predictions, axis=1)]
    y_true = valid_df['label'].values
    labels = np.unique(valid_df['label'].values)
    cm = confusion_matrix(y_true, classes, labels=labels)
    df = pd.DataFrame(cm, columns=labels, index=labels)
    df.to_csv(os.path.join(params['output_path'], 'confusion.csv'), index_label='index')
    print(df)
    return df


def main_train(params):
    print(params)
    chekpoints_path = os.path.join(params['output_path'], NAME + '_weights')
    os.makedirs(chekpoints_path, exist_ok=True)
    batch_size = params['batch_size']
    n = params['sample_size']

    train_data, validate_data = load_train_data(params['audio_path'], params['validation_list_path'])
    assert len(train_data) != 0
    assert len(validate_data) != 0

    silence_data = get_silence(train_data)

    if params['sample']:
        print("Get small sample")
        train_data, validate_data = get_sample_data(train_data, validate_data, n)

    train_gen = train_generator(train_data, silence_data, batch_size, n=n)
    validate_gen = valid_generator(validate_data, silence_data, batch_size)
    model = build()

    train(model, train_gen, validate_gen, dict(
        epochs=params['epochs'],
        batch_size=batch_size,
        tensorboard_dir=os.path.join(params['tensorboard_root'], NAME),
        chekpoints_path=chekpoints_path,
        steps_per_epoch=n * len(LABELS) / batch_size,
        validation_steps=int(np.ceil(validate_data.shape[0] / batch_size))
    ))
