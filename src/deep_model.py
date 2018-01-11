# -*- coding: utf-8 -*-
import os

from keras import Input, metrics
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras.engine import Model, Layer
from keras.layers import Conv1D, Dense, K, regularizers, BatchNormalization, Activation, MaxPooling1D, \
    GlobalAveragePooling1D
from keras.optimizers import Adam

from classifier import Classifier


class Deep1DClassifier(Classifier):
    def __init__(self, L, labels) -> None:
        super().__init__(L, labels)
        self._model = self._build()
        self._name = "Deep1DClassifier"

    def _layer(self, x: Layer, filters: int, kernel_size: int, strides: int):
        x = Conv1D(filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding='same',
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=regularizers.l2(l=0.0001))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    def _build(self):
        inputs = Input(shape=(self.L, 1), dtype='float32')
        x = inputs

        x = self._layer(x, filters=48, kernel_size=160, strides=2)
        x = MaxPooling1D(pool_size=4, strides=None)(x)

        x = self._layer(x, filters=64, kernel_size=3, strides=1)
        x = self._layer(x, filters=64, kernel_size=3, strides=1)
        x = self._layer(x, filters=64, kernel_size=3, strides=1)
        x = self._layer(x, filters=64, kernel_size=3, strides=1)
        x = MaxPooling1D(pool_size=4, strides=None)(x)

        x = self._layer(x, filters=128, kernel_size=3, strides=1)
        x = self._layer(x, filters=128, kernel_size=3, strides=1)
        x = self._layer(x, filters=128, kernel_size=3, strides=1)
        x = self._layer(x, filters=128, kernel_size=3, strides=1)
        x = MaxPooling1D(pool_size=4, strides=None)(x)

        x = self._layer(x, filters=256, kernel_size=3, strides=1)
        x = self._layer(x, filters=256, kernel_size=3, strides=1)
        x = self._layer(x, filters=256, kernel_size=3, strides=1)
        x = self._layer(x, filters=256, kernel_size=3, strides=1)
        x = MaxPooling1D(pool_size=4, strides=None)(x)

        x = self._layer(x, filters=512, kernel_size=3, strides=1)
        x = self._layer(x, filters=512, kernel_size=3, strides=1)
        x = self._layer(x, filters=512, kernel_size=3, strides=1)
        x = self._layer(x, filters=512, kernel_size=3, strides=1)
        x = GlobalAveragePooling1D()(x)

        x = Dense(len(self.labels), activation='softmax')(x)

        return Model(inputs, x, name=self.name)

    def train(self, train_gen, validation_gen, params):
        print(params)
        self._model.summary()
        self._model.compile(optimizer=Adam(),
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
            ), ReduceLROnPlateau(
                monitor='val_acc',
                factor=0.5,
                patience=10,
                min_lr=0.0001,
                verbose=1)
            ])

    @property
    def name(self):
        return self._name
