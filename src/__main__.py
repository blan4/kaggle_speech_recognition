# -*- coding: utf-8 -*-
import sys
from datetime import datetime

from m_02_sound_2d import train, build
from sound_processing import process_wav_file_to_2d
from sound_reader import read_and_process
from submission import main_predict, main_confusion_matrix
from train import main_train


def main(args):
    if len(args) != 3:
        print("Do `python3 src [local, gcloud, floyd] [predict, train]`")
        exit(-1)
    params = {}
    process_sound = read_and_process(process_wav_file_to_2d)
    if args[1] == 'floyd':
        print("FLOYD ENV")
        params = {'data_path': '/data',
                  'output_path': '/output',
                  'audio_path': '/data/*/*wav',
                  'validation_list_path': '/data/validation_list.txt',
                  'tensorboard_root': '/output',
                  'sample': False,
                  'sample_size': 1000,
                  'epochs': 20,
                  'batch_size': 64,
                  'submission_path': './output/submission{}.csv'.format(datetime.now()),
                  'model_path': './output/weights-improvement-20-0.76.hdf5',
                  'test_path': '???',
                  'batch_size_pred': 64,
                  'process_wav': process_sound
                  }
    elif args[1] == 'gcloud':
        print("GCLOUD ENV")
        params = {'data_path': '/mnt/data/speech/',
                  'output_path': './output',
                  'audio_path': '/mnt/data/speech/train/audio/*/*wav',
                  'validation_list_path': '/mnt/data/speech/train/validation_list.txt',
                  'tensorboard_root': './output',
                  'sample': False,
                  'sample_size': 1000,
                  'epochs': 40,
                  'batch_size': 64,
                  'submission_path': './submissions/submission{}.csv'.format(datetime.now()),
                  'model_path': './output/BaselineSpeech_weights/weights-improvement-20-0.76.hdf5',
                  'test_path': '/mnt/data/speech/test/audio/*wav',
                  'batch_size_pred': 64,
                  'process_wav': process_sound
                  }
    elif sys.argv[1] == 'local':
        print("DEV ENV")
        params = {'data_path': './data',
                  'output_path': './output',
                  'audio_path': './data/train/audio/*/*wav',
                  'validation_list_path': './data/train/validation_list.txt',
                  'tensorboard_root': '/tmp/tensorflow/',
                  'sample': True,
                  'sample_size': 40,
                  'epochs': 10,
                  'batch_size': 8,
                  'submission_path': './submissions/submission{}.csv'.format(datetime.now()),
                  'model_path': './weights/weights-improvement-20-0.76.hdf5',
                  'test_path': './data/test/audio/*wav',
                  'batch_size_pred': 1,
                  'process_wav': process_sound
                  }
    else:
        print("Do `python3 src [local, gcloud, floyd] [predict, train]`")
        exit(-1)

    if args[2] == 'predict':
        main_predict(params)
    elif args[2] == 'train':
        main_train(params, build(), train)
    elif args[2] == 'confusion':
        main_confusion_matrix(params)
    else:
        print("Do `python3 src [local, gcloud, floyd] [predict, train, confusion]`")
        exit(-1)


if __name__ == '__main__':
    main(sys.argv)
