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
from torch import from_numpy
import nltk
import warnings
warnings.filterwarnings('ignore')

from get_data import * 
from train_fn import * 

class PlainEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, dropout=0.2):
        #以英文为例，vocab_size=5493, hidden_size=100, dropout=0.2
        super(PlainEncoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size) #这里的hidden_size为embedding_dim：一个单词的维度 
        self.rnn = nn.GRU(embedding_size, hidden_size, batch_first = True)      
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths): 
        '''需要把最后一个hidden state取出来，需要知道长度，因为句子长度不一样
        x: english sequence 
        lengths: batch里每个句子的长度
        
        x: [batch_size, seq_len] => embed 
        => embedded:[batch_size, seq_len, embedding_size] => pack_padded 
        => packed_embedded:[batch_size, batch_size*seq_len, embedding_size] => rnn 
        => packed_out:[batch_size, batch_size*seq_len, hidden_size]; => pad_packed
           hid:[1, batch_size, hidden_size] 
        => out: [batch_size, seq_len, hidden_size], hid[[-1]] [1, batch_size, hidden_size]
        '''
        sorted_lengths, sorted_idx = lengths.sort(0, descending=True) # 按照长度降序排序
        x_sorted = x[sorted_idx.long()] # [batch_size, seq_len]
        embedded = self.dropout(self.embed(x_sorted)) # [batch_size, seq_len, embedding_size]
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded,  # embedding后的句子
                                                            sorted_lengths.long().cpu().data.numpy(),  # 句子长度
                                                            batch_first=True)
        # packed_embedded [batch_size*seq_len, embedding_size]
        packed_out, hid = self.rnn(packed_embedded) # [batch_size*seq_len, embedding_size], [batch_size, embedding_size]
        # pad_packed： padding操作
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first = True) # [seq_len, embedding_size]
        _, original_idx = sorted_idx.sort(0, descending=False)  # 按照原始idx升序排列
        out = out[original_idx.long()].contiguous() # torch.contiguous()方法首先拷贝了一份张量在内存中的地址，然后将地址按照形状改变后的张量的语义进行排列。
        hid = hid[:, original_idx.long()].contiguous()
        return out, hid[[-1]]
    
class PlainDecoder(nn.Module):
    '''基于中文输入和英文encoder的最后一个隐藏层的输出，得到最终的翻译语句
    y [batch_size, seq_len] => embed
    => y_embed: [batch_size, seq_len, embed_size] => pack_padded
    => packed_seq: [batch_size, batch_size*seq_len, embed_size] => rnn
    => packed_out: [batch_size, batch_size*seq_len, hidden_size] => pad_packed
       hidden: [1, batch_size, hidden_size]
    => unpacked_out: [batch_size, seq_len, hidden_size] => log_softmax
    => output: [batch_size, seq_len, cn_vocab_size]
       hidden: [1, batch_size, hidden_size]
    '''  
    def __init__(self, vocab_size, embedding_size, hidden_size, dropout = 0.2):
        super(PlainDecoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.GRU(embedding_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)  
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, y, y_lengths, hid):
        sorted_lengths, sorted_idx = y_lengths.sort(0, descending=True)
        y_sorted, hid = y[sorted_idx.long()], hid[:, sorted_idx.long()] 
        y_embed = self.dropout(self.embed(y_sorted))  # [batch_size, seq_len, embed_size]
        packed_seq = nn.utils.rnn.pack_padded_sequence(y_embed, 
                                                       sorted_lengths.long().cpu().data.numpy(), 
                                                       batch_first=True)
        packed_out, hid = self.rnn(packed_seq, hid) #加上隐藏层
        unpacked_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        _, original_idx = sorted_idx.sort(0, descending=False)
        output_seq = unpacked_out[original_idx.long()].contiguous()
        hid = hid[:, original_idx.long()].contiguous()
        output = F.log_softmax(self.out(output_seq), -1)
        return output, hid  

class PlainSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(PlainSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, x, x_lengths, y, y_lengths):
        encoder_out, encoder_hid = self.encoder(x, x_lengths)
        output, decoder_hid = self.decoder(y = y, y_lengths = y_lengths, 
                                        hid = encoder_hid)
        return output, None

    def translate(self, x, x_lengths, y, max_length = 10):
        preds = []
        attns = []
        batch_size = x.shape[0]
        encoder_out, encoder_hid = self.encoder(x, x_lengths)
        for i in range(max_length):
            output, decoder_hid = self.decoder(y = y,
                                    y_lengths = torch.ones(batch_size).long().to(y.device),
                                    hid = encoder_hid) 
            #刚开始循环bos作为模型的首个输入单词，后续更新y，下个预测单词的输入是上个输出单词
            y = output.max(2)[1].view(batch_size, 1)
            preds.append(y)
        return torch.cat(preds, 1), None

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        input = input.contiguous().view(-1, input.size(2))
        target = target.contiguous().view(-1, 1)
        mask = mask.contiguous().view(-1, 1)
        output = -input.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)
        return output


def main():
    # 导入数据并分词
    train_file = "nmt/en-cn/train.txt"
    dev_file = "nmt/en-cn/dev.txt"
    train_en, train_cn = load_data(train_file)
    dev_en, dev_cn = load_data(dev_file)

    # 构建单词表
    en_dict, en_total_words = build_dict(train_en) # en_dict: word_to_idx
    inv_en_dict = {v: k for k, v in en_dict.items()} # idx_to word
    cn_dict, cn_total_words = build_dict(train_cn)
    inv_cn_dict = {v: k for k, v in cn_dict.items()}

    # convert text[i] to sequence[i]
    train_en_seq, train_cn_seq = encode(train_en, train_cn, en_dict, cn_dict)
    dev_en_seq, dev_cn_seq = encode(dev_en, dev_cn, en_dict, cn_dict)

    # 把全部句子分成batch
    batch_size = 32
    train_data = gen_examples(train_en_seq, train_cn_seq, batch_size)
    dev_data = gen_examples(dev_en_seq, dev_cn_seq, batch_size)

    # 训练模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    en_vocab_size = len(en_dict)
    cn_vocab_size = len(cn_dict)
    embedding_size = 100
    hidden_size = 64
    num_epochs= 1
    dropout = 0.1
    encoder = PlainEncoder(vocab_size = en_vocab_size, 
                      embedding_size = embedding_size,
                      hidden_size=hidden_size,
                      dropout=dropout)

    decoder = PlainDecoder(vocab_size=cn_vocab_size, 
                        embedding_size = embedding_size,
                        hidden_size=hidden_size,
                        dropout=dropout)

    model = PlainSeq2Seq(encoder, decoder)
    model = model.to(device)
    loss_fn = LanguageModelCriterion().to(device)
    optimizer = torch.optim.Adam(model.parameters())

    random.shuffle(train_data)
    model = train(model, train_data, 
        num_epochs= num_epochs, 
        device = device, 
        loss_fn = loss_fn,
        optimizer = optimizer,
        valid_data = dev_data)

    for i in range(100,120):
        translate_dev(model, device, dev_en_seq, inv_en_dict, dev_cn_seq, inv_cn_dict,cn_dict,i)
        print()


if __name__ == '__main__':
    main()