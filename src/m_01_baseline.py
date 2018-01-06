# -*- coding: utf-8 -*-
import os

from keras import Input, metrics
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.engine import Model
from keras.layers import Conv1D, Flatten, Dense, Dropout, K, BatchNormalization, MaxPooling1D, AveragePooling1D
from keras.optimizers import SGD

from consts import L, LABELS

NAME = "BaselineSpeech"


def build():
    inputs = Input(shape=(L, 1))  # 16000
    x = inputs

    x = Conv1D(filters=8, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    # x = Conv1D(filters=8, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2, padding='same')(x)

    x = Conv1D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    # x = Conv1D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2, padding='same')(x)
    x = Dropout(0.2)(x)

    x = Conv1D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    # x = Conv1D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2, padding='same')(x)
    x = Dropout(0.2)(x)

    x = Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    # x = Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2, padding='same')(x)
    x = Dropout(0.2)(x)

    x = Conv1D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    # x = Conv1D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2, padding='same')(x)
    x = Dropout(0.2)(x)

    x = Conv1D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    # x = Conv1D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2, padding='same')(x)
    x = Dropout(0.2)(x)

    x = Conv1D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    # x = Conv1D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2, padding='same')(x)
    x = Dropout(0.2)(x)

    x = Conv1D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    # x = Conv1D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2, padding='same')(x)
    x = Dropout(0.2)(x)

    x = Conv1D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    # x = Conv1D(filters=512, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2, padding='same')(x)
    x = Dropout(0.2)(x)

    # x = Conv1D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    # x = Conv1D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu')(x)

    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    # x = Dropout(0.5)(x)
    # x = Dense(256, activation='relu')(x)
    x = Dense(len(LABELS), activation='softmax')(x)

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
            batch_size=params['batch_size']
        ), ModelCheckpoint(
            os.path.join(params['chekpoints_path'],
                         "weights-improvement-{epoch:02d}-{val_categorical_accuracy:.2f}.hdf5"),
            monitor='val_categorical_accuracy',
            verbose=1,
            save_best_only=True,
            mode='auto'
        )])
