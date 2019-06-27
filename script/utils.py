# coding: utf-8


def load_corpus(path):
    corpus = []
    with open(path + ".txt", 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            corpus.append(line.strip())
            line = f.readline()
    return corpus


def save_corpus(corpus, path):
    with open(path + ".txt", 'w', encoding='utf-8') as f:
        for data in corpus:
            f.write(" ".join([str(value) for value in data]) + "\n")
