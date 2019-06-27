# coding: utf-8


import torch
import torch.nn as nn
import numpy as np
import visdom


def train(train_task, criterion):
    train_task.mode()
    loss_list = []
    pairs = train_task.pairs

    for inp, tar in pairs:
        train_task.optim_zero_grad()
        out = train_task.model(inp)
        loss = criterion(out, torch.LongTensor([tar]))
        loss.backward()
        train_task.optim_step()
        loss_list.append(loss.item())

    return sum(loss_list)/len(loss_list)


def validate(validate_task, criterion):
    validate_task.mode()
    loss_list = []
    pairs = validate_task.pairs

    for inp, tar in pairs:
        out = validate_task.model(inp)
        loss = criterion(out, torch.LongTensor([tar]))
        loss_list.append(loss.item())

    return sum(loss_list)/len(loss_list)


def train_validate_loop(train_task, validate_task, log_path):
    vis = visdom.Visdom()
    criterion = nn.NLLLoss()
    epoch = 1

    input = torch.randn(3, 5, requires_grad=True)
    target = torch.empty(3, dtype=torch.long).random_(5)

    print(input, target)

    while True:
        print("TRAIN (" + str(epoch-1) + "-" + str(epoch) + "): ", end="")
        train_loss = train(train_task, criterion)
        print(train_loss)

        with open(log_path + ".txt", 'a', encoding='utf-8') as f:
            f.write("TRAIN (" + str(epoch-1) + "-" + str(epoch) + "): " + str(train_loss) + "\n")

        vis.line(X=np.array([epoch]), Y=np.array([train_loss]), win=log_path, name="train loss", update='append')

        if epoch % validate_task.interval == 0:
            print("VALIDATE (" + str(epoch) + "): ", end="")
            validate_loss = validate(validate_task, criterion)
            print(validate_loss)

            with open(log_path + ".txt", 'a', encoding='utf-8') as f:
                f.write("VALIDATE (" + str(epoch) + "): " + str(validate_loss) + "\n")

            vis.line(X=np.array([epoch]), Y=np.array([validate_loss]), win=log_path, name="validate loss", update='append')

        epoch += 1


def train_start(train_task, validate_task, log_path):
    try:
        train_validate_loop(train_task, validate_task, log_path)
    except KeyboardInterrupt:
        return
