# -*- coding: utf-8 -*-
import os

from keras import Input, metrics
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.engine import Model
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, K
from keras.optimizers import SGD

from consts import L, LABELS, output_path
from data_loader import load_train_data, data_generator

NAME = "BaselineSpeech"


def build():
    inputs = Input(shape=(L, 1))  # 16000
    x = Conv1D(filters=128, kernel_size=4, strides=2, padding='same', activation='relu')(inputs)  # 8000x128
    x = MaxPooling1D(pool_size=4, padding='same')(x)  # 2000x128
    x = Conv1D(filters=256, kernel_size=4, strides=1, padding='same', activation='relu')(x)  # 2000x256
    x = MaxPooling1D(pool_size=4, padding='same')(x)  # 500x256
    x = Conv1D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu')(x)  # 500x512
    x = MaxPooling1D(pool_size=4, padding='same')(x)  # 125x512
    x = Conv1D(filters=1024, kernel_size=3, strides=1, padding='same', activation='relu')(x)  # 125x1024
    x = MaxPooling1D(pool_size=4, padding='same')(x)  # 32x1024
    x = Conv1D(filters=1024, kernel_size=3, strides=1, padding='same', activation='relu')(x)  # 32x1024
    x = MaxPooling1D(pool_size=4, padding='same')(x)  # 32x1024
    x = Conv1D(filters=1024, kernel_size=3, strides=2, padding='same', activation='relu')(x)  # 4x1024
    x = Dropout(0.5)(x)
    x = Flatten()(x)  # 32768
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)  # 256
    x = Dense(len(LABELS), activation='sigmoid')(x)  # 12

    return Model(inputs, x, name=NAME)


def train(model, train_gen, validation_gen, params):
    model.summary()
    model.compile(optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                  loss=K.categorical_crossentropy,
                  metrics=[metrics.categorical_accuracy])

    return model.fit_generator(
        generator=train_gen(),
        steps_per_epoch=params['steps_per_epoch'],
        epochs=params['epochs'],
        validation_data=validation_gen(),
        validation_steps=params['validation_steps'],
        callbacks=[TensorBoard(
            log_dir=params['tensorboard_dir'],
            write_images=True,
            batch_size=params['batch_size']
        ), ModelCheckpoint(
            os.path.join(params['chekpoints_path'], "weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"),
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            mode='auto'
        )])


def main(data_path, tensorboard_root='/tmp/tensorflow/'):
    chekpoints_path = os.path.join(output_path, NAME + '_models')
    os.makedirs(chekpoints_path, exist_ok=True)
    batch_size = 64

    train_data, validate_data = load_train_data(data_path)
    assert len(train_data) != 0
    assert len(validate_data) != 0

    train_gen = data_generator(train_data, batch_size, shuffle=True)
    validate_gen = data_generator(validate_data, batch_size, shuffle=True)
    model = build()

    train(model, train_gen, validate_gen, dict(
        epochs=10,
        batch_size=batch_size,
        tensorboard_dir=os.path.join(tensorboard_root, NAME),
        chekpoints_path=chekpoints_path,
        steps_per_epoch=len(train_data) // batch_size,
        validation_steps=len(validate_data) // batch_size
    ))
