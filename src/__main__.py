# -*- coding: utf-8 -*-
import sys
from datetime import datetime

from m_01_baseline import train, build
from submission import main_predict, main_confusion_matrix
from train import main_train

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Do `python3 src [local, gcloud, floyd] [predict, train]`")
        exit(-1)
    params = {}
    if sys.argv[1] == 'floyd':
        print("FLOYD ENV")
        params = {'data_path': '/data',
                  'output_path': '/output',
                  'audio_path': '/data/*/*wav',
                  'validation_list_path': '/data/validation_list.txt',
                  'tensorboard_root': '/output',
                  'sample': False,
                  'sample_size': 2000,
                  'epochs': 20,
                  'batch_size': 64,
                  'submission_path': './output/submission{}.csv'.format(datetime.now()),
                  'model_path': './output/weights-improvement-20-0.76.hdf5',
                  'test_path': '???',
                  'batch_size_pred': 64
                  }
    elif sys.argv[1] == 'gcloud':
        print("GCLOUD ENV")
        params = {'data_path': '/mnt/data/speech/',
                  'output_path': './output',
                  'audio_path': '/mnt/data/speech/train/audio/*/*wav',
                  'validation_list_path': '/mnt/data/speech/train/validation_list.txt',
                  'tensorboard_root': './output',
                  'sample': False,
                  'sample_size': 2000,
                  'epochs': 40,
                  'batch_size': 64,
                  'submission_path': './submissions/submission{}.csv'.format(datetime.now()),
                  'model_path': './output/BaselineSpeech_weights/weights-improvement-20-0.76.hdf5',
                  'test_path': '/mnt/data/speech/test/audio/*wav',
                  'batch_size_pred': 64
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
                  'batch_size_pred': 1
                  }
    else:
        print("Do `python3 src [local, gcloud, floyd] [predict, train]`")
        exit(-1)

    if sys.argv[2] == 'predict':
        main_predict(params)
    elif sys.argv[2] == 'train':
        main_train(params, build(), train)
    elif sys.argv[2] == 'confusion':
        main_confusion_matrix(params)
    else:
        print("Do `python3 src [local, gcloud, floyd] [predict, train]`")
        exit(-1)
