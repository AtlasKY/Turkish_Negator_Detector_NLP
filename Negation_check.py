

import sys
from gensim.corpora import WikiCorpus
from gensim.models.word2vec import Word2Vec, LineSentence
from gensim.models import KeyedVectors
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_model(model_name):

    model = Word2Vec.load(model_name)
    wv = model.wv
    return model, wv


if __name__ == '__main__':
    model, wv = load_model("wiki_tr_w2v.model")
    print(wv.most_similar(positive='alamam'))
    embedding = nn.Embedding.from_pretrained(model)
    input = torch.LongTensor([1])
    embedding(input)
