# -*- coding: utf-8 -*-

import random

import numpy as np
from pysndfx import AudioEffectsChain


class SoundEffects:
    """
    Add some sound effects:
        - speed up
        - speed down
        - reverberation
        - tremolo
        - high shelf
        - low shelf
    Also apply sound normalization
    """

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.sample_size = self.sample_rate

    def call(self, inputs):
        return self._apply_effect(inputs)

    def _speed_up(self, chain: AudioEffectsChain):
        np.random.rand()
        return chain.speed(factor=np.random.randint(11, 17) / 10)

    def _speed_down(self, chain: AudioEffectsChain):
        return chain.speed(factor=np.random.randint(7, 10) / 10)

    def _reverb(self, chain: AudioEffectsChain):
        return chain.reverb()

    def _tremolo(self, chain: AudioEffectsChain):
        return chain \
            .tremolo(freq=np.random.randint(self.sample_rate // 10, self.sample_rate)) \
            .tremolo(freq=np.random.randint(self.sample_rate // 10, self.sample_rate)) \
            .tremolo(freq=np.random.randint(self.sample_rate // 10, self.sample_rate))

    def _highshelf(self, chain: AudioEffectsChain):
        return chain.highshelf(frequency=3000)

    def _lowshelf(self, chain: AudioEffectsChain):
        return chain.lowshelf(frequency=300)

    def _fix_size(self, x):
        rem_len = self.sample_size - len(x)
        if rem_len > 0:
            return np.append(x, np.random.uniform(0.0, 0.1, size=rem_len))
        elif rem_len < 0:
            i = np.random.randint(0, -rem_len)
            return x[i:i + self.sample_size]
        else:
            return x

    def _apply_effect(self, x):
        effect = AudioEffectsChain()
        effects = [self._lowshelf,
                   self._highshelf,
                   self._speed_up,
                   self._speed_down,
                   self._tremolo,
                   self._reverb]
        for apply in random.sample(effects, np.random.randint(0, 5)):
            effect = apply(effect)

        x = effect(x, sample_in=self.sample_rate, sample_out=self.sample_rate)
        return self._fix_size(x)
