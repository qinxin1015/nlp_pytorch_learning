#coding:utf-8
"""
author: qinxin
data:2021.12.6
word2vec 介绍了两种训练词向量的模型，skip-gram和cbow

skip-gram: 使用中心词预测周围词

cbow: 使用周围词预测中心词

这个函数基于pytorch实现skip-gram, 并保存训练得到的词向量，embedding_weights
"""

import torch
import torch.nn as nn

import random
import pandas as pd
import numpy as np


'''difine parameters'''
C = 3
K = 100
MAX_VOCAB_SIZE = 30000
EMBEDDING_SIZE = 100  # 一般而言，2**EMBEDDING_SIZE > MAX_VOCAB_SIZE
NUM_WORKERS = 4
NUM_EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.1

DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")

from collections import Counter

class BuildVocab(object):
    """build vocabilary based on text, 

    Args:
        :param: text: build vocabilary based on text
        :param: MAX_VOCAB_SIZE: the size of vocabilary 
        :param: EMBEDDING_SIZE: embedding vocabilary to embedding_weights of EMBEDDING_SIZE
        :return:
    """
    def __init__(self, MAX_VOCAB_SIZE,EMBEDDING_SIZE):
        self.VOCAB_SIZE = MAX_VOCAB_SIZE
        self.EMBEDDING_SIZE = EMBEDDING_SIZE
        self.vocab = dict()
        self.idx_to_word = []
        self.word_to_idx = dict()

    def word_tokenize(self, text):
        return text.split()

    def build_vocab(self,text):
        text = [w for w in self.word_tokenize(text.lower())]
        self.vocab = dict(Counter(text).most_common(self.VOCAB_SIZE - 1)) # 统计常见词的词频
        self.vocab["<unk>"] = len(text) - np.sum(list(self.vocab.values()))  # 不常见的词都设为unk

        self.idx_to_word = [word for word in self.vocab.keys()]
        self.word_to_idx = {word:idx for idx, word in enumerate(self.idx_to_word)}
        self.VOCAB_SIZE = len(self.idx_to_word)

    def word_freqs(self):
        if len(self.vocab) == 0:
            raise ValueError("vocab no words")

        word_counts = np.array([count for count in self.vocab.values()])
        word_freqs = word_counts / np.sum(word_counts)
        word_freqs = word_freqs ** (3./4.)
        word_freqs = word_freqs / np.sum(word_freqs)
        return word_freqs

from torch.utils.data import Dataset, DataLoader

class WordEmbeddingDataset(Dataset):
    """difine the dataset to training model 
    
    Arg:
        text: context
        word_to_idx: dict, {word:idx} from vocab
        idx_to_word: list, word from vocab
        word_freqs: array, word freqs in vocab
        C: the window size of context form text
        K: numbers of multiple when sampling negative samples
        return: center_word, pos_words, neg_words
    """
    def __init__(self, text, VOCAB_SIZE, word_to_idx, idx_to_word, word_freqs, C, K):
        super(WordEmbeddingDataset,self).__init__()
        self.text_encoded = [word_to_idx.get(word, VOCAB_SIZE-1) for word in text]
        self.text_encoded = torch.LongTensor(self.text_encoded)
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.word_freqs = torch.Tensor(word_freqs)
        self.C = C
        self.K = K

    def __len__(self):
        return len(self.text_encoded)

    def __getitem__(self, idx):
        center_word = self.text_encoded[idx]
        pos_idx = list(range(idx-self.C, idx)) + list(range(idx+1, idx+self.C+1))
        pos_words = self.text_encoded[pos_idx]
        neg_words = torch.multinomial(self.word_freqs, self.K*pos_words.shape[0])
        return center_word, pos_words, neg_words


import torch.nn as nn
import torch.nn.functional as F

class EmbeddingModel(nn.Module):
    """Skip-gram model 
    """
    def __init__(self, vocab_size, embedding_size):
        super(EmbeddingModel,self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.in_embed = nn.Embedding(self.vocab_size, self.embedding_size)
        self.out_embed = nn.Embedding(self.vocab_size, self.embedding_size)
        # 模型参数初始化
        init_weight = 0.5 / self.embedding_size
        self.in_embed.weight.data.uniform_(-init_weight, init_weight)
        self.out_embed.weight.data.uniform_(-init_weight, init_weight)

    def forward(self, center_word, pos_words, neg_words):
        center_embedding = self.in_embed(center_word) # [batch_size, embedding_size]
        pos_embedding = self.out_embed(pos_words)     # [batch_size, 2C, embedding_size]
        neg_embedding = self.out_embed(neg_words)     # [batch_size, 2C*K, embedding_size]
        # unsqueeze(dim)  在第dim个维度增加一个维度
        center_unsqueeze = center_embedding.unsqueeze(2)      # [batch_size, embedding_size, 1]
        log_pos = torch.bmm(pos_embedding, center_unsqueeze)  # [batch_size, 2C, 1]
        log_neg = torch.bmm(-neg_embedding, center_unsqueeze) # [batch_size, 2C*K, 1]
        # squeeze() 去掉维度是1的维度
        log_pos = log_pos.squeeze()  # [batch_size, 2C]
        log_neg = log_neg.squeeze()  # [batch_size, 2C*K]

        log_pos = F.logsigmoid(log_pos).sum(1)
        log_neg = F.logsigmoid(log_neg).sum(1)

        loss = log_pos + log_neg

        return -loss

    def input_embedding(self,):
        return self.in_embed.weight.data.cpu().numpy()

    def output_embedding(self,):
        return self.out_embed.weight.data.cpu().numpy()


from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine

class ModelEvaluate(object):
    def __init__(self, file, embedding_weights, word_to_idx, idx_to_word):
        self.embedding_weights = embedding_weights
        self.data = self.read_file(file)
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word

    def read_file(self, file):
        if file.endwith(".csv"):
            data = pd.read_csv(file, sep = ",")
        else:
            data = pd.read_csv(file, sep = "\t")
        return data

    def word_correlation(self,):
        data = self.data
        model_similarity = []
        human_similarity = []

        for i in data.iloc[:,0:2].index():
            w1, w2 = data.iloc[i,0], data.iloc[i,1]
            if w1 not in self.word_to_idx or w2 not in self.word_to_idx:
                print("{} or {} not in vocab".format(w1,w2))
                continue
            else:
                w1_idx, w2_idx = self.word_to_idx[w1],self.word_to_idx[w2]
                w1_embed, w2_embed = self.embedding_weights[[w1_idx]], self.embedding_weights[[w2_idx]]
                model_similarity.append(float(cosine_similarity(w1_embed, w2_embed)))
                human_similarity.append(data.iloc[i,2])

        corr = spearmanr(model_similarity, human_similarity)
        return corr

    def find_nearest(self, word, topn = 10):
        """返回余弦相似度最小的topn个词
        """
        idx = self.word_to_idx[word]
        word_embed = self.embedding_weights[idx]
        cos_dist = np.array([cosine(e, embed) for e in embedding_weights])

        return [self.idx_to_word[i] for i in cos_dist.argsort[:topn]]


from torch.optim import SGD, Adam

def model_train(model, dataloader, EMBEDDING_SIZE, LEARNING_RATE,NUM_EPOCHS, vnum = 1000):

    optimizer = SGD(model.parameters(), lr = LEARNING_RATE)
    for e in range(NUM_EPOCHS):
        for i, (center,pos,neg) in enumerate(dataloader):
            center = torch.LongTensor(center).to(DEVICE)
            pos = torch.LongTensor(pos).to(DEVICE)
            neg = torch.LongTensor(neg).to(DEVICE)

            optimizer.zero_grad()
            loss = model(center, pos, neg).mean()
            loss.backward()
            optimizer.step()

            if i % vnum == 0:
                print("epoch: {}, iter: {}, loss: {}".format(e, i, loss.item()))

    torch.save(model.state_dict(), "embedding_weights-{}.th".format(EMBEDDING_SIZE))


def main():

    train_file = "./data/text8/text8.train.txt"
    with open(train_file, "r") as fin:
        train_data = fin.read()

    vocab = BuildVocab(MAX_VOCAB_SIZE,EMBEDDING_SIZE)
    vocab.build_vocab(train_data) # 得到 vocab,word_to_idx,idx_to_word
    word_freqs = vocab.word_freqs()

    word_to_idx, idx_to_word = vocab.word_to_idx, vocab.idx_to_word
    VOCAB_SIZE = vocab.VOCAB_SIZE

    dataset = WordEmbeddingDataset(train_data, VOCAB_SIZE, word_to_idx, idx_to_word, word_freqs, C, K)
    dataloader = DataLoader(dataset = dataset,
                            batch_size = BATCH_SIZE,
                            num_workers = NUM_WORKERS,
                            shuffle = True)


    model = EmbeddingModel(MAX_VOCAB_SIZE, EMBEDDING_SIZE)
    model = model.to(DEVICE)
    model_train(model, dataloader, EMBEDDING_SIZE, LEARNING_RATE, NUM_EPOCHS, vnum = 1000)

    model.load_state_dict(torch.load("embedding_weights-{}.th".format(EMBEDDING_SIZE)))




if __name__ == '__main__':
    main()