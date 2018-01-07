# -*- coding: utf-8 -*-
import os

from keras import Input, metrics
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.engine import Model
from keras.layers import Conv1D, Flatten, Dense, Dropout, K, MaxPooling1D
from keras.optimizers import SGD, RMSprop, Adam

from classifier import Classifier


class Classifier1D(Classifier):
    def __init__(self, L, labels) -> None:
        super().__init__(L, labels)
        self._model = self._build()
        self._name = "BaselineSpeech"

    def _build(self):
        inputs = Input(shape=(self.L, 1), dtype='float32')  # 16000
        x = inputs

        x = Conv1D(filters=8, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        # x = Conv1D(filters=8, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        x = MaxPooling1D(pool_size=2, padding='same')(x)

        x = Conv1D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        # x = Conv1D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        x = MaxPooling1D(pool_size=2, padding='same')(x)
        # x = Dropout(0.2)(x)

        x = Conv1D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        # x = Conv1D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        x = MaxPooling1D(pool_size=2, padding='same')(x)
        # x = Dropout(0.2)(x)

        x = Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        # x = Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        x = MaxPooling1D(pool_size=2, padding='same')(x)
        # x = Dropout(0.2)(x)

        x = Conv1D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        # x = Conv1D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        x = MaxPooling1D(pool_size=2, padding='same')(x)
        # x = Dropout(0.2)(x)

        x = Conv1D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        # x = Conv1D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        x = MaxPooling1D(pool_size=2, padding='same')(x)
        # x = Dropout(0.2)(x)

        x = Conv1D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        # x = Conv1D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        x = MaxPooling1D(pool_size=2, padding='same')(x)
        # x = Dropout(0.2)(x)

        x = Conv1D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        # x = Conv1D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        x = MaxPooling1D(pool_size=2, padding='same')(x)
        # x = Dropout(0.2)(x)

        x = Conv1D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        # x = Conv1D(filters=512, kernel_size=3, strides=2, padding='same', activation='relu')(x)
        x = MaxPooling1D(pool_size=2, padding='same')(x)
        # x = Dropout(0.2)(x)

        # x = Conv1D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        # x = Conv1D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu')(x)

        x = Flatten()(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        # x = Dropout(0.5)(x)
        # x = Dense(256, activation='relu')(x)
        x = Dense(len(self.labels), activation='softmax')(x)

        return Model(inputs, x, name=self.name)

    def train(self, train_gen, validation_gen, params):
        print(params)
        self._model.summary()
        self._model.compile(optimizer=RMSprop(),
                            loss=K.categorical_crossentropy,
                            metrics=[metrics.categorical_accuracy])

        return self._model.fit_generator(
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

    @property
    def name(self):
        return self._name
