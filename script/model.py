# coding: utf-8


import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, model_param):
        super(Model, self).__init__()
        self.model_param = model_param

        layer = []
        for i in range(len(model_param)-1):
            layer.append(nn.Linear(model_param[i], model_param[i+1]))
            if i < len(model_param)-2:
                layer.append(nn.ReLU())
        layer.append(nn.LogSoftmax(dim=0))

        self.net = nn.Sequential(*layer)

    def forward(self, x):
        y = self.net(x)
        return y.view(1, self.model_param[-1])

    def save(self, path):
        torch.save(self.state_dict(), path + ".pth")

    def load(self, path):
        self.load_state_dict(torch.load(path + ".pth"))
