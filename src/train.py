# -*- coding: utf-8 -*-
import os
from datetime import datetime

import numpy as np

import sound_processing as sp
from consts import L
from data_loader import load_train_data, train_generator, valid_generator, get_silence, get_sample_data
from sound_chain import SoundChain
from sound_reader import SimpleWavFileReader


def main_train(params, model, train):
    name = "{}--{}".format(model.name, int(datetime.now().timestamp()))
    print(params)
    chekpoints_path = os.path.join(params['output_path'], name + '_weights')
    os.makedirs(chekpoints_path, exist_ok=True)
    batch_size = params['batch_size']
    n = params['sample_size']

    train_data, validate_data = load_train_data(params['audio_path'], params['validation_list_path'])
    assert len(train_data) != 0
    assert len(validate_data) != 0

    wav_reader = SimpleWavFileReader(L)
    silence_data = get_silence(train_data, wav_reader)

    train_sound_chain = SoundChain(
        SimpleWavFileReader(L),
        sp.AdjustLenWavProcessor(silence_data, L, L),
        # sp.AddNoiseWavProcessor(silence_data, L, L, 20),
        # sp.ShiftWavProcessor(silence_data, L, L),
        sp.EmphasisWavProcessor(silence_data, L, L, 0.97),
        sp.NormalizeWavProcessor(silence_data, L, L),
        sp.ReshapeWavProcessor(silence_data, L, L),
    )

    valid_sound_chain = SoundChain(
        SimpleWavFileReader(L),
        sp.AdjustLenWavProcessor(silence_data, L, L),
        sp.EmphasisWavProcessor(silence_data, L, L, 0.97),
        sp.NormalizeWavProcessor(silence_data, L, L),
        sp.ReshapeWavProcessor(silence_data, L, L),
    )

    if params['sample']:
        print("Get small sample")
        train_data, validate_data = get_sample_data(train_data, validate_data, n)

    train_gen = train_generator(train_data, batch_size, train_sound_chain, n=n)
    validate_gen = valid_generator(validate_data, batch_size, valid_sound_chain, True)

    train(model, train_gen, validate_gen, dict(
        epochs=params['epochs'],
        batch_size=batch_size,
        tensorboard_dir=os.path.join(params['tensorboard_root'], name),
        chekpoints_path=chekpoints_path,
        steps_per_epoch=n * len(LABELS) / batch_size,
        validation_steps=int(np.ceil(validate_data.shape[0] / batch_size))
    ))
