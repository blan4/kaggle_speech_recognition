# -*- coding: utf-8 -*-
import os

from keras import Input, metrics
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.engine import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, K, BatchNormalization
from keras.optimizers import SGD

from consts import LABELS, strides, frame_size

NAME = "BaselineSpeech"


def build():
    inputs = Input(shape=(frame_size, strides, 1))  # 320x160x1
    x = inputs

    x = MaxPooling2D(pool_size=(2, 1), padding='same')(x)  # 160x160x1
    x = Conv2D(64, kernel_size=(6, 3), strides=1, padding='same', activation='relu')(x)  # 160x160x64
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)  # 80x80x64
    x = Conv2D(128, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)  # 80x80x128
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)  # 40x40x128
    x = Conv2D(256, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)  # 40x40x256
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)  # 20x20x256
    x = Conv2D(256, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)  # 20x20x256
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)  # 10x10x256
    x = Conv2D(512, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)  # 10x10x512
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)  # 5x5x512
    x = Conv2D(512, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)  # 5x5x512
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)  # 3x3x512
    x = BatchNormalization()(x)

    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(len(LABELS), activation='sigmoid')(x)

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
