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



def load_data(in_file):
    cn = []
    en = []
    num_examples = 0
    with open(in_file, 'r', encoding = "utf-8") as f:
        for line in f:
            line = line.strip().split("\t") # 分词后用逗号隔开
            en.append(["BOS"] + nltk.word_tokenize(line[0].lower()) + ["EOS"]) # BOS:句子起始标志 EOS:结束标志
            cn.append(["BOS"] + [c for c in line[1]] + ["EOS"]) # 中文一个一个字分词，可以尝试用分词器分词
            
    return en, cn


def build_dict(sentences, max_words = 50000):
    UNK_IDX = 0
    PAD_IDX = 1
    word_count = Counter()
    for sentence in sentences:
        for s in sentence:
            word_count[s] += 1  # word_count这里应该是个字典
    ls = word_count.most_common(max_words) 
    print(len(ls)) #train_en：5491
    total_words = len(ls) + 2
    word_dict = {w[0]: index+2 for index, w in enumerate(ls)}
    #加的2是留给"unk"和"pad",转换成字典格式。
    word_dict["UNK"] = UNK_IDX
    word_dict["PAD"] = PAD_IDX
    return word_dict, total_words


def encode(en_sentences, cn_sentences, en_dict, cn_dict, sort_by_len = True):
    '''
        Encode the sequences. 
    '''
    length = len(en_sentences)
    out_en_sentences = [[en_dict.get(w, 0) for w in sent] for sent in en_sentences]
    out_cn_sentences = [[cn_dict.get(w, 0) for w in sent] for sent in cn_sentences]

    # sort sentences by english lengths
    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))
       
    # 把中文和英文按照同样的顺序排序
    if sort_by_len:
        sorted_index = len_argsort(out_en_sentences)
        out_en_sentences = [out_en_sentences[i] for i in sorted_index]
        out_cn_sentences = [out_cn_sentences[i] for i in sorted_index]
        
    return out_en_sentences, out_cn_sentences


def get_minibatches(n, minibatch_size, shuffle=True):
    idx_list = np.arange(0, n, minibatch_size) # [0, 1, ..., n-1]
    if shuffle:
        np.random.shuffle(idx_list)            # 打乱数据
    minibatches = []
    for idx in idx_list:
        minibatches.append(np.arange(idx, min(idx + minibatch_size, n)))
    return minibatches


def main():
    # 导入数据并分词
    train_file = "nmt/en-cn/train.txt"
    dev_file = "nmt/en-cn/dev.txt"
    train_en, train_cn = load_data(train_file)
    print(train_en[0],train_cn[0])
    dev_en, dev_cn = load_data(dev_file)

    # 构建单词表
    en_dict, en_total_words = build_dict(train_en)
    cn_dict, cn_total_words = build_dict(train_cn)
    inv_en_dict = {v: k for k, v in en_dict.items()}
    inv_cn_dict = {v: k for k, v in cn_dict.items()}

    # 把单词全部转变成数字
    train_en, train_cn = encode(train_en, train_cn, en_dict, cn_dict)
    dev_en, dev_cn = encode(dev_en, dev_cn, en_dict, cn_dict)




if __name__ == '__main__':
    main()
