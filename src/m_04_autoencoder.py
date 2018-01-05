# -*- coding: utf-8 -*-
import os

from keras import Input
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.engine import Model
from keras.layers import Conv1D, MaxPooling1D, UpSampling1D, K, Dropout, BatchNormalization
from keras.optimizers import Adadelta

from consts import L

NAME = "AutoEncoderSpeech"


def build():
    inputs = Input(shape=(L, 1))  # 16000
    x = inputs

    # Encoder
    x = Conv1D(filters=16, kernel_size=15, strides=1, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=4, padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv1D(filters=32, kernel_size=5, strides=1, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=4, padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=4, padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2, padding='same')(x)
    x = BatchNormalization()(x)

    x = Dropout(rate=0.25)(x)

    # Decoder
    x = Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = UpSampling1D(size=2)(x)
    x = BatchNormalization()(x)
    x = Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = UpSampling1D(size=4)(x)
    x = BatchNormalization()(x)
    x = Conv1D(filters=32, kernel_size=5, strides=1, padding='same', activation='relu')(x)
    x = UpSampling1D(size=4)(x)
    x = BatchNormalization()(x)
    x = Conv1D(filters=16, kernel_size=15, strides=1, padding='same', activation='relu')(x)
    x = UpSampling1D(size=4)(x)
    x = BatchNormalization()(x)
    x = Conv1D(filters=1, kernel_size=15, strides=1, padding='same', activation='sigmoid')(x)

    return Model(inputs, x, name=NAME)


def train(model, train_gen, validation_gen, params):
    print(params)
    model.summary()
    model.compile(optimizer=Adadelta(),
                  loss=K.binary_crossentropy)

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
                         "weights-improvement-{epoch:02d}-{val_binary_crossentropy:.2f}.hdf5"),
            monitor='val_binary_crossentropy',
            verbose=1,
            save_best_only=True,
            mode='auto'
        )])
