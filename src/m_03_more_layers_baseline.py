# -*- coding: utf-8 -*-
import os

from keras import Input, metrics
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.engine import Model
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, K, BatchNormalization
from keras.optimizers import SGD

from consts import L, LABELS

NAME = "BaselineSpeech"


def build():
    inputs = Input(shape=(L, 1))  # 16000
    x = inputs
    conf = [
        ['c', 1, 5, 128],  # 16000x128
        ['p', 5],  # 3200x128
        ['c', 1, 5, 128],  # 3200x128
        ['p', 5],  # 640x128
        ['c', 1, 4, 256],  # 640x128
        ['p', 5],  # 128x256
        ['c', 1, 3, 256],  # 128x256
        ['p', 4],  # 32x256
        ['c', 1, 3, 512],  # 32x512
        ['p', 2],  # 16x512
        ['c', 1, 3, 512],  # 16x512
        ['p', 2],  # 8x512
        ['c', 1, 3, 1024],  # 8x512
        ['p', 2],  # 4x512
        ['c', 1, 3, 1024],  # 4x1024
    ]

    for layer in conf:
        if layer[0] == 'c':
            x = Conv1D(filters=layer[3], kernel_size=layer[2], strides=layer[1], padding='same', activation='relu')(x)
        elif layer[0] == 'p':
            x = MaxPooling1D(pool_size=layer[1], padding='same')(x)
        else:
            print("Unknown layer")

    x = Flatten()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
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
