# -*- coding: utf-8 -*-
import os

from keras import Input, metrics
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.engine import Model
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, K
from keras.optimizers import SGD

from consts import L, LABELS
from data_loader import load_train_data, data_generator

NAME = "BaselineSpeech"


def build():
    inputs = Input(shape=(L, 1))  # 16000
    x = Conv1D(filters=128, kernel_size=3, strides=2, padding='same', activation='relu')(inputs)  # 8000x128
    x = MaxPooling1D(pool_size=2, padding='same')(x)  # 4000x128
    x = Conv1D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu')(x)  # 4000x128
    x = MaxPooling1D(pool_size=2, padding='same')(x)  # 2000x128
    x = Conv1D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu')(x)  # 2000x256
    x = MaxPooling1D(pool_size=2, padding='same')(x)  # 1000x256
    x = Conv1D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu')(x)  # 1000x256
    x = MaxPooling1D(pool_size=2, padding='same')(x)  # 500x256
    x = Conv1D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu')(x)  # 500x512
    x = MaxPooling1D(pool_size=3, padding='same')(x)  # 166x512
    x = Conv1D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu')(x)  # 166x512
    x = MaxPooling1D(pool_size=3, padding='same')(x)  # 55x512
    x = Conv1D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu')(x)  # 55x512
    x = MaxPooling1D(pool_size=3, padding='same')(x)  # 18x512
    x = Conv1D(filters=512, kernel_size=3, strides=2, padding='same', activation='relu')(x)  # 9x512
    x = Flatten()(x)  # 4608
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)  # 256
    x = Dense(len(LABELS), activation='sigmoid')(x)  # 12

    return Model(inputs, x, name=NAME)


def train(model, train_gen, validation_gen, epochs=10, batch_size=64):
    model.summary()
    model.compile(optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                  loss=K.categorical_crossentropy,
                  metrics=[metrics.categorical_accuracy])

    return model.fit_generator(
        generator=train_gen(),
        steps_per_epoch=60000 / batch_size,
        epochs=epochs,
        shuffle=True,
        validation_data=validation_gen(),
        validation_steps=7000 / batch_size,
        callbacks=[TensorBoard(
            log_dir="/tmp/tensorflow/{}".format(NAME),
            write_images=True,
            histogram_freq=5,
            batch_size=batch_size
        ), ModelCheckpoint(
            "output/" + NAME + "/weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5",
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            mode='auto'
        )])


def main(data_path):
    os.makedirs(os.path.join("output", NAME))

    train_data, validate = load_train_data(data_path)
    train_gen = data_generator(train_data)
    validate_gen = data_generator(validate)
    model = build()
    train(model, train_gen, validate_gen, epochs=10, batch_size=64)
