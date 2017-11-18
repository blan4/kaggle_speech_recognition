# -*- coding: utf-8 -*-

L = 16000
N = 2000
LABELS = 'yes no up down left right on off stop go silence unknown'.split()
id2name = {i: name for i, name in enumerate(LABELS)}
name2id = {name: i for i, name in id2name.items()}

id_to_label = {0: 'down',
               1: 'left',
               2: 'right',
               4: 'unknown',
               5: 'off',
               6: 'on',
               7: 'go',
               8: 'stop',
               9: 'up',
               10: 'yes'}

label_to_id = {'down': 0,
               'go': 7,
               'left': 1,
               'no': 5,
               'off': 5,
               'on': 6,
               'right': 2,
               'stop': 8,
               'unknown': 4,
               'up': 9,
               'yes': 10}