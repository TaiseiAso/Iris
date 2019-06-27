# coding: utf-8


# 必要なモジュールをインポート
from script import *
import yaml
import torch


torch.backends.cudnn.benchmark = True


config = yaml.load(stream=open("config/config.yml", 'rt', encoding='utf-8'), Loader=yaml.SafeLoader)
path = config['path']
model_param = config['model_param']
train = config['train']
validate = config['validate']


corpus_train_input = load_corpus(path['corpus'] + train['corpus_name']['input'])
corpus_train_target = load_corpus(path['corpus'] + train['corpus_name']['target'])
corpus_validate_input = load_corpus(path['corpus'] + validate['corpus_name']['input'])
corpus_validate_target = load_corpus(path['corpus'] + validate['corpus_name']['target'])


model = Model(model_param)


train_task = TrainTask(corpus_train_input, corpus_train_target, model, train['learning_rate'])
validate_task = ValidateTask(corpus_validate_input, corpus_validate_target, model, validate['interval'])


train_start(train_task, validate_task, path['save'] + train['log_name'])


model.save(path['save'] + train['model_name'])
