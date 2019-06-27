# coding: utf-8


import torch
from torch import optim


class TrainTask:
    def __init__(self, corpus_input, corpus_target, model, learning_rate):
        self.model = model
        self.optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        self.pairs = [
            [torch.FloatTensor([float(value) for value in inp.split()]), int(tar)]
            for inp, tar in zip(corpus_input, corpus_target)
        ]

    def mode(self):
        self.model.train()

    def optim_zero_grad(self):
        self.optimizer.zero_grad()

    def optim_step(self):
        self.optimizer.step()


class ValidateTask:
    def __init__(self, corpus_input, corpus_target, model, interval):
        self.model = model
        self.interval = interval
        self.pairs = [
            [torch.FloatTensor([float(value) for value in inp.split()]), int(tar)]
            for inp, tar in zip(corpus_input, corpus_target)
        ]

    def mode(self):
        self.model.eval()


class TestTask:
    def __init__(self, corpus_input, corpus_target, model):
        self.model = model
        self.pairs = [
            [torch.FloatTensor([float(value) for value in inp.split()]), int(tar)]
            for inp, tar in zip(corpus_input, corpus_target)
        ]

    def mode(self):
        self.model.eval()
