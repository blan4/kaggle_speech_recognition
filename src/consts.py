# -*- coding: utf-8 -*-

L = 16000
LABELS = set('yes no up down left right on off stop go silence unknown'.split())
id2name = {i: name for i, name in enumerate(LABELS)}
name2id = {name: i for i, name in id2name.items()}


audio_path = 'audio/*/*wav'
validation_list_path = 'validation_list.txt'
output_path = 'output'
