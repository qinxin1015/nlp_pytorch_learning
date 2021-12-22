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
from get_data import * 
from train_fn import * 

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, enc_hidden_size, dec_hidden_size, dropout=0.2):
        super(Encoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, enc_hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(enc_hidden_size * 2, dec_hidden_size)

    def forward(self, x, lengths):
        sorted_len, sorted_idx = lengths.sort(0, descending=True)
        x_sorted = x[sorted_idx.long()]
        embedded = self.dropout(self.embed(x_sorted))
        
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_len.long().cpu().data.numpy(), batch_first=True)
        packed_out, hid = self.rnn(packed_embedded)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        _, original_idx = sorted_idx.sort(0, descending=False)
        out = out[original_idx.long()].contiguous()
        hid = hid[:, original_idx.long()].contiguous()
        
        hid = torch.cat([hid[-2], hid[-1]], dim=1)
        hid = torch.tanh(self.fc(hid)).unsqueeze(0)
        return out, hid

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, enc_hidden_size, dec_hidden_size, dropout=0.2):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(enc_hidden_size, dec_hidden_size)
        self.rnn = nn.GRU(embed_size, enc_hidden_size, batch_first = True)
        self.out = nn.Linear(dec_hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def create_mask(self, x_len, y_len):
        device = x_len.device
        max_x_len = x_len.max()
        max_y_len = y_len.max()
        x_mask = torch.arange(max_x_len, device=x_len.device)[None, :] < x_len[:, None]
        y_mask = torch.arange(max_y_len, device=x_len.device)[None, :] < y_len[:, None]
        mask = (~ x_mask[:, :, None] * y_mask[:, None, :]).byte()
        return mask
        
    def forward(self, ctx, ctx_lengths, y, y_lengths, hid):
        sorted_len, sorted_idx = y_lengths.sort(0, descending=True)
        y_sorted = y[sorted_idx.long()]
        hid = hid[:, sorted_idx.long()]
        
        y_sorted = self.dropout(self.embed(y_sorted)) # batch_size, output_length, embed_size

        packed_seq = nn.utils.rnn.pack_padded_sequence(y_sorted, sorted_len.long().cpu().data.numpy(), batch_first=True)
        out, hid = self.rnn(packed_seq, hid)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        _, original_idx = sorted_idx.sort(0, descending=False)
        output_seq = unpacked[original_idx.long()].contiguous()
        hid = hid[:, original_idx.long()].contiguous()
        mask = self.create_mask(y_lengths, ctx_lengths)
        # code.interact(local=locals())
        output, attn = self.attention(output_seq, ctx, mask)
        output = F.log_softmax(self.out(output), -1)
        return output, hid, attn

class Attention(nn.Module):
    """ Luong Attention
    output: batch_size, output_len, dec_hidden_size
    context: batch_size, context_len, enc_hidden_size
    """
    def __init__(self, enc_hidden_size, dec_hidden_size):
        super(Attention, self).__init__()

        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size

        self.linear_in = nn.Linear(enc_hidden_size*2, dec_hidden_size, bias=False)
        self.linear_out = nn.Linear(enc_hidden_size*2 + dec_hidden_size, dec_hidden_size)
        
    def forward(self, output, context, mask):
        batch_size = output.size(0)
        output_len = output.size(1)
        input_len = context.size(1)
        
        context_in = self.linear_in(context.view(batch_size*input_len, -1)).view(                batch_size, input_len, -1) # batch_size, output_len, dec_hidden_size
        attn = torch.bmm(output, context_in.transpose(1,2)) # batch_size, output_len, context_len

        attn.data.masked_fill(mask, -1e6)
        attn = F.softmax(attn, dim=2) # batch_size, output_len, context_len
        context = torch.bmm(attn, context) # batch_size, output_len, enc_hidden_size
        output = torch.cat((context, output), dim=2) # batch_size, output_len, hidden_size*2
        output = output.view(batch_size*output_len, -1)
        output = torch.tanh(self.linear_out(output))
        output = output.view(batch_size, output_len, -1)
        return output, attn

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, x, x_lengths, y, y_lengths):
        encoder_out, hid = self.encoder(x, x_lengths)
        output, hid, attn = self.decoder(ctx=encoder_out, 
                    ctx_lengths=x_lengths,
                    y=y,
                    y_lengths=y_lengths,
                    hid=hid)
        return output, attn
    
    def translate(self, x, x_lengths, y, max_length=100):
        encoder_out, hid = self.encoder(x, x_lengths)
        preds = []
        batch_size = x.shape[0]
        attns = []
        for i in range(max_length):
            output, hid, attn = self.decoder(ctx=encoder_out, 
                                        ctx_lengths=x_lengths,
                                        y=y,
                                        y_lengths=torch.ones(batch_size).long().to(y.device),
                                        hid=hid)
            preds.append(output.max(2)[1].view(batch_size, 1))
            attns.append(attn)
        return torch.cat(preds, 1), torch.cat(attns, 1)

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
    batch_size = 64
    train_data = gen_examples(train_en_seq, train_cn_seq, batch_size)
    dev_data = gen_examples(dev_en_seq, dev_cn_seq, batch_size)

    # 训练模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    en_vocab_size = len(en_dict)
    cn_vocab_size = len(cn_dict)
    embed_size = hidden_size = 100
    dropout = 0.5

    encoder = Encoder(vocab_size=en_vocab_size, 
                      embed_size=embed_size, 
                      enc_hidden_size=hidden_size,
                      dec_hidden_size=hidden_size,
                      dropout=dropout)

    decoder = Decoder(vocab_size=cn_vocab_size, 
                      embed_size=embed_size, 
                      enc_hidden_size=hidden_size,
                      dec_hidden_size=hidden_size,
                      dropout=dropout)
    model = Seq2Seq(encoder, decoder)
    model = model.to(device)
    loss_fn = LanguageModelCriterion().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    random.shuffle(train_data)
    train(model, train_data, 
        num_epochs= 30, 
        device = device, 
        loss_fn = loss_fn,
        optimizer = optimizer,
        valid_data = dev_data)


    # translate_dev
    for i in range(100,120):
        translate_dev(i)
        print()



if __name__ == '__main__':
    main()
