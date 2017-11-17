# -*- coding: utf-8 -*-
import sys

from m_01_baseline import main

audio_path = 'audio/*/*wav'
validation_list_path = 'validation_list.txt'
output_path = 'output'

if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'prod':
        print("PRODUCTION ENV")
        params = {'data_path': '/data',
                  'output_path': '/output',
                  'audio_path': '/data/*/*wav',
                  'validation_list_path': '/data/validation_list.txt',
                  'tensorboard_root': '/output',
                  'sample': False,
                  'sample_size': 2000,
                  'epochs': 20,
                  'batch_size': 64
                  }
    else:
        print("DEV ENV")
        params = {'data_path': './data',
                  'output_path': './output',
                  'audio_path': './data/train/audio/*/*wav',
                  'validation_list_path': './data/train/validation_list.txt',
                  'tensorboard_root': '/tmp/tensorflow/',
                  'sample': True,
                  'sample_size': 40,
                  'epochs': 10,
                  'batch_size': 8
                  }

    main(params)
