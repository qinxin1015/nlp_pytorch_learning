#coding:utf-8
import os
import sys
import math
from collections import Counter #计数器
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
import warnings
warnings.filterwarnings('ignore')


def load_data(in_file):
    cn = []
    en = []
    num_examples = 0
    with open(in_file, 'r',encoding="utf-8") as f:
        for line in f:
            #print(line) #Anyone can do that.   任何人都可以做到。
            line = line.strip().split("\t") #分词后用逗号隔开
            #print(line) # ['Anyone can do that.', '任何人都可以做到。']
            en.append(["BOS"] + nltk.word_tokenize(line[0].lower()) + ["EOS"])
            #BOS:beginning of sequence EOS:end of
            # split chinese sentence into characters
            cn.append(["BOS"] + [c for c in line[1]] + ["EOS"])
            #中文一个一个字分词，可以尝试用分词器分词
    return en, cn

def build_dict(sentences, max_words=50000):
    UNK_IDX = 0
    PAD_IDX = 1
    word_count = Counter()
    for sentence in sentences:
        for s in sentence:
            word_count[s] += 1  # word_count这里应该是个字典
    ls = word_count.most_common(max_words) 
    total_words = len(ls) + 2  #train_en：5491 + 2
    #加的2是留给"unk"和"pad"
    #ls = [('BOS', 14533), ('EOS', 14533), ('.', 12521), ('i', 4045), .......
    word_dict = {word[0]: idx+2 for idx, word in enumerate(ls)}
    word_dict["UNK"] = UNK_IDX
    word_dict["PAD"] = PAD_IDX
    return word_dict, total_words

def encode(en_sentences, cn_sentences, en_dict, cn_dict, sort_by_len=True):
    '''Encode the sequences. 
    en_sentences=[['BOS', 'anyone', 'can', 'do', 'that', '.', 'EOS'],....
    out_en_sentences=[[2, 328, 43, 14, 28, 4, 3], ....
    sorted_index=[63, 1544, 1917, 2650, 3998, 6240, 6294, 6703, ....
    out_en_sentences=[[2, 475, 4, 3], [2, 1318, 126, 3], [2, 1707, 126, 3], ...... 按照长度排好序的seq
    '''
    length = len(en_sentences)
    en_sequences = [[en_dict.get(w, 0) for w in sent] for sent in en_sentences] #.get(w, 0)，返回w对应的值，没有就为0.
    cn_sequences = [[cn_dict.get(w, 0) for w in sent] for sent in cn_sentences]
    # sort sentences by english lengths
    def len_argsort(seq):
        """返回句子长度的list"""
        return sorted(range(len(seq)), key = lambda x: len(seq[x]))
    #sorted()排序,key参数可以自定义规则，按seq[x]的长度排序，seq[0]为第一句话长度
    # 把中文和英文按照同样的顺序排序
    if sort_by_len:
        sorted_index = len_argsort(en_sentences)
        out_en_sequences = [en_sequences[i] for i in sorted_index]
        out_cn_sequences = [cn_sequences[i] for i in sorted_index]
    return out_en_sequences, out_cn_sequences

def get_minibatches(n, batch_size, shuffle = True):
    '''返回 batch index'''
    idx_list = np.arange(0, n, batch_size) # [0, 1, ..., n-1]
    if shuffle:
        np.random.shuffle(idx_list) #打乱数据
    minibatches = []
    for idx in idx_list:
        minibatches.append(np.arange(idx, min(idx + batch_size, n)))
    return minibatches

def prepare_data(seqs):
    '''use 0 padding seq '''
    lengths = [len(seq) for seq in seqs] # 每个batch里语句的长度统计出来
    n_samples = len(seqs) #一个batch有多少语句
    max_len = np.max(lengths) #取出最长的的语句长度，后面用这个做padding基准
    x = np.zeros((n_samples, max_len)).astype('int32')  #先初始化全零矩阵，后面依次赋值
    x_lengths = np.array(lengths).astype("int32")
    #这里看下面的输入语句发现英文句子长度都一样，中文句子长短不一。说明英文句子是特征，中文句子是标签。
    for idx, seq in enumerate(seqs):
        x[idx, :lengths[idx]] = seq
    return x, x_lengths # x_mask

def gen_examples(en_sentences, cn_sentences, batch_size):
    minibatches = get_minibatches(len(en_sentences), batch_size)
    all_ex = []
    for minibatch in minibatches:
        mb_en_sentences = [en_sentences[t] for t in minibatch]
        mb_cn_sentences = [cn_sentences[t] for t in minibatch]
        # 返回的维度为：mb_x=(64 * 最大句子长度）,mb_x_len=最大句子长度
        mb_x, mb_x_len = prepare_data(mb_en_sentences)
        mb_y, mb_y_len = prepare_data(mb_cn_sentences)
        all_ex.append((mb_x, mb_x_len, mb_y, mb_y_len))
    # 这里把所有batch数据集合到一起。 [英文句子，英文长度，中文句子翻译，中文句子长度]
    return all_ex