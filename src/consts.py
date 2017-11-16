# -*- coding: utf-8 -*-

import numpy as np

L = 16000
LABELS = set('yes no up down left right on off stop go silence unknown'.split())
id2name = {i: name for i, name in enumerate(LABELS)}
name2id = {name: i for i, name in id2name.items()}


def id_to_one_hot(i):
    vec = np.zeros(len(LABELS))
    vec[i] = 1
    return vec
