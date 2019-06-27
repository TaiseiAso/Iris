# coding: utf-8


# 必要なモジュールをインポート
from script import *
import yaml
import torch


torch.backends.cudnn.benchmark = True


config = yaml.load(stream=open("config/config.yml", 'rt', encoding='utf-8'), Loader=yaml.SafeLoader)
path = config['path']
model_param = config['model_param']
test = config['test']


corpus_test_input = load_corpus(path['corpus'] + test['corpus_name']['input'])
corpus_test_target = load_corpus(path['corpus'] + test['corpus_name']['target'])


model = Model(model_param)
model.load(path['save'] + test['model_name'])


test_task = TestTask(corpus_test_input, corpus_test_target, model)


test_start(test_task, path['save'] + test['log_name'])
