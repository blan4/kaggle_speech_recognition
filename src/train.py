# -*- coding: utf-8 -*-
import os

import numpy as np

from consts import LABELS
from data_loader import load_train_data, train_generator, valid_generator, get_silence, get_sample_data


def main_train(params, model, train):
    name = model.name
    print(params)
    chekpoints_path = os.path.join(params['output_path'], name + '_weights')
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

    train_gen = train_generator(train_data, silence_data, batch_size, params['process_wav'], n=n)
    validate_gen = valid_generator(validate_data, silence_data, batch_size, params['process_wav'])

    train(model, train_gen, validate_gen, dict(
        epochs=params['epochs'],
        batch_size=batch_size,
        tensorboard_dir=os.path.join(params['tensorboard_root'], name),
        chekpoints_path=chekpoints_path,
        steps_per_epoch=n * len(LABELS) / batch_size,
        validation_steps=int(np.ceil(validate_data.shape[0] / batch_size))
    ))
