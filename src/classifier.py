# -*- coding: utf-8 -*-
from abc import abstractmethod


class Classifier:
    def __init__(self, L, labels) -> None:
        self.L = L
        self.labels = labels
        self._name = "AbstractClassifier"

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def train(self, train_gen, validation_gen, params):
        pass
