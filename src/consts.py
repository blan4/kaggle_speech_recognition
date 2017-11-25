# -*- coding: utf-8 -*-

L = 16000
N = 2000
LABELS = 'yes no up down left right on off stop go silence unknown'.split()
id2name = {i: name for i, name in enumerate(LABELS)}
name2id = {name: i for i, name in id2name.items()}

print(LABELS)
print(id2name)
print(name2id)

ms_to_s = 1000.0
frame_size = int((20 / ms_to_s) * L)
stride_size = int((10 / ms_to_s) * L)
strides = int(L / stride_size)
