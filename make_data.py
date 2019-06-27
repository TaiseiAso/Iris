# coding: utf-8


from script import *
import yaml
import random
from sklearn.datasets import load_iris


config = yaml.load(stream=open("config/config.yml", 'rt', encoding='utf-8'), Loader=yaml.SafeLoader)
corpus_path = config['path']['corpus']
data_split = config['data_split']
train_corpus_name = config['train']['corpus_name']
validate_corpus_name = config['validate']['corpus_name']
test_corpus_name = config['test']['corpus_name']


iris = load_iris()


inp_list, tar_list = [], []
for inp, tar in zip(iris.data, iris.target):
    inp_list.append(inp.tolist())
    tar_list.append([tar])

pair_list = list(zip(inp_list, tar_list))
random.shuffle(pair_list)
inp_list, tar_list = zip(*pair_list)


size = len(inp_list)

all = data_split['train'] + data_split['validate'] + data_split['test']
train_rate = data_split['train']/all
validate_rate = data_split['validate']/all

train_size = int(size*train_rate)
validate_size = int(size*validate_rate)

train_inp_list = inp_list[:train_size]
train_tar_list = tar_list[:train_size]

validate_inp_list = inp_list[train_size:train_size+validate_size]
validate_tar_list = tar_list[train_size:train_size+validate_size]

test_inp_list = inp_list[train_size+validate_size:]
test_tar_list = tar_list[train_size+validate_size:]


save_corpus(train_inp_list, corpus_path + train_corpus_name['input'])
save_corpus(train_tar_list, corpus_path + train_corpus_name['target'])
save_corpus(validate_inp_list, corpus_path + validate_corpus_name['input'])
save_corpus(validate_tar_list, corpus_path + validate_corpus_name['target'])
save_corpus(test_inp_list, corpus_path + test_corpus_name['input'])
save_corpus(test_tar_list, corpus_path + test_corpus_name['target'])
