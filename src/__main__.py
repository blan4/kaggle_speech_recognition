# -*- coding: utf-8 -*-
import sys
from datetime import datetime

from m_01_baseline import Classifier1D
from submission import main_predict, main_confusion_matrix
from train import main_train


def main(args):
    help_str = "Do `python3 src [local, gcloud, floyd, devbox] [predict, train, confusion]`"

    if len(args) != 3:
        print(help_str)
        exit(-1)
    params = {}
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
                  'batch_size_pred': 64
                  }
    elif args[1] == 'gcloud':
        print("GCLOUD ENV")
        params = {'data_path': '/mnt/data/speech/',
                  'output_path': './output',
                  'audio_path': '/mnt/data/speech/train/audio/*/*wav',
                  'validation_list_path': '/mnt/data/speech/train/validation_list.txt',
                  'tensorboard_root': './output',
                  'sample': False,
                  'sample_size': 2000,
                  'epochs': 60,
                  'batch_size': 64,
                  'submission_path': './submissions/submission{}.csv'.format(datetime.now()),
                  'model_path': './output/BaselineSpeech_weights/weights-improvement-20-0.76.hdf5',
                  'test_path': '/mnt/data/speech/test/audio/*wav',
                  'batch_size_pred': 64
                  }
    elif args[1] == 'devbox':
        print("DEVBOX ENV")
        params = {'data_path': '/home/ilya/Data/speech/',
                  'output_path': '/home/ilya/Data/speech/out/output',
                  'audio_path': '/home/ilya/Data/speech/train/audio/*/*wav',
                  'validation_list_path': '/home/ilya/Data/speech/train/validation_list.txt',
                  'tensorboard_root': '/home/ilya/Data/speech/out/output',
                  'sample': False,
                  'sample_size': 2000,
                  'epochs': 200,
                  'batch_size': 64,
                  'submission_path': '/home/ilya/Data/speech/out/submissions/submission{}.csv'.format(datetime.now()),
                  'model_path': '???',
                  'test_path': '/home/ilya/Data/speech/test/audio/*wav',
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
        print(help_str)
        exit(-1)

    if args[2] == 'predict':
        main_predict(params)
    elif args[2] == 'train':
        main_train(params, Classifier1D)
    elif args[2] == 'confusion':
        main_confusion_matrix(params)
    else:
        print(help_str)
        exit(-1)


if __name__ == '__main__':
    main(sys.argv)
