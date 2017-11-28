# -*- coding: utf-8 -*-

ALL_LABELS = 'bed bird cat dog down eight five four go happy house left marvin nine no off on one right seven sheila six stop three tree two up wow yes zero silence unknown'.split()
TASK_LABELS = 'yes no up down left right on off stop go silence unknown'.split()

L = 16000
LABELS = TASK_LABELS
id2name = {i: name for i, name in enumerate(LABELS)}
name2id = {name: i for i, name in id2name.items()}

print(LABELS)
print(id2name)
print(name2id)

ms_to_s = 1000.0
frame_size = int((20 / ms_to_s) * L)  # 320
stride_size = int((6.25 / ms_to_s) * L)  # 100
strides = int(L / stride_size)  # 160
