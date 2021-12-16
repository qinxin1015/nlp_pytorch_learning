import torch
from torchtext.legacy import data, datasets
# from torchtext import data,datasets

EMBEDDING_SIZE = 100
OUTPUT_SIZE = 1
BATCH_SIZE = 64
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


import torch.nn as nn
import torch.nn.functional as F
# 定义模型
class WordAVGModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, output_size, pad_idx):
        super(WordAVGModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size, padding_idx = pad_idx)
        self.linear = nn.Linear(embedding_size, output_size)
    
    def forward(self, text):
        embedded = self.embed(text) # [seq_len, batch_size, embedding_size]
        embedded = embedded.permute(1,0,2) # [batch_size, seq_len, embedding_size]
        pooled = F.avg_pool2d(embedded, kernel_size=(embedded.shape[1], 1)).squeeze() # [batch_size, embedding_size]
        out = self.linear(pooled)
        return out

class LSTMModel(nn.Module):
    def __init__(self, vocab_size,embedding_size, pad_idx, hidden_size,output_size,dropout):
        super(LSTMModel,self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size, padding_idx = pad_idx)
        self.lstm = nn.LSTM(embedding_size, hidden_size, 
                            # num_layers = 1,
                            bidirectional = False, 
                            dropout = dropout)
        self.linear = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self,text):
        embedded = self.dropout(self.embed(text)) # [seq_len, batch_size, 1] => [seq_len, batch_size, embedding_size]
        output, (hidden, C) = self.lstm(embedded) # [seq_len, batch_size, embedding_size] 
                                                  # => output: [seq_len, batch_size, hidden_size]
                                                  # => hidden/C: [1(num_layers), batch_size, hidden_size]

        hidden = self.dropout(hidden.squeeze()) # [1, batch_size, hidden_size] => [batch_size, hidden_size]
        out = self.linear(hidden) # [batch_size, hidden_size] => [batch_size, 1]
        return out  # [batch_size, 1] 

class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size,embedding_size,pad_idx,hidden_size,output_size, rnn_layers,dropout):
        super(BiLSTMModel,self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size, padding_idx = pad_idx)
        self.lstm = nn.LSTM(embedding_size, hidden_size, 
                            num_layers = rnn_layers,
                            bidirectional = True, 
                            dropout = dropout)
        self.linear = nn.Linear(2*hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self,text):
        embedded = self.dropout(self.embed(text)) # [seq_len, batch_size, 1] => [seq_len, batch_size, embedding_size]
        output, (hidden, C) = self.lstm(embedded) # [seq_len, batch_size, embedding_size] 
                                                  # => output: [seq_len, batch_size, 2 * hidden_size]
                                                  # => hidden/C: [2 * num_layers, batch_size, hidden_size]
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)# [2 * num_layers, batch_size, hidden_size] => 2 * [batch_size, hidden_size] => [batch_size, 2 * hidden_size]                                     
        hidden = self.dropout(hidden) 
        out = self.linear(hidden) # # [batch_size, 2 * hidden_size] => [batch_size, 1]
        return out  # [batch_size, 1] 

class CNNModel(nn.Module):
    '''一个conv层'''
    def __init__(self, vocab_size, embedding_size, n_filters, 
                 filter_sizes, output_size, dropout, pad_idx):

        super(CNNModel,self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size, padding_idx = pad_idx)
        self.convs = nn.Conv2d(in_channels = 1, out_channels = n_filters, 
                               kernel_size = (filter_sizes, embedding_size))                                    
        self.linear = nn.Linear(n_filters, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self,text):
        # [seq_len, batch_size] => [batch_size, seq_len]
        text = text.permute(1, 0) 
        # [batch_size, seq_len] => [batch_size, seq_len, embedding_size]
        embedded = self.embed(text) 
        # [batch_size, seq_len, embedding_size] => [batch_size, 1, seq_len, embedding_size]
        embedded = embedded.unsqueeze(1) 
        # [batch_size, 1, seq_len, embedding_size] => [batch_size, n_filters, seq_len-filter_sizes+1, 1]
        #                         conved.squeeze() => [batch_size, n_filters, seq_len-filter_sizes+1]
        conved = F.relu(self.convs(embedded)).squeeze(3)  
        # [batch_size, n_filters, seq_len-filter_sizes+1] => [batch_size, n_filters, 1]
        #                                pooled.squeeze() => [batch_size, n_filters]
        pooled = F.max_pool1d(conved, conved.shape[2]).squeeze(2) 
        # [batch_size, n_filters] => [batch_size, output_size]
        out = self.linear(pooled)
        return out

class CNNsModel(nn.Module):
    '''多个conv层'''
    def __init__(self, vocab_size, embedding_size, n_filters, 
                 filter_sizes, output_size, dropout, pad_idx):
        super(CNNsModel,self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size, padding_idx = pad_idx)
        self.convs = nn.ModuleList([
                            nn.Conv2d(in_channels = 1, out_channels = n_filters, 
                               kernel_size = (filter_size, embedding_size)) 
                            for filter_size in filter_sizes
                            ])   # len(filter_sizes) 个 convs()                                
        self.linear = nn.Linear(len(filter_sizes) * n_filters, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self,text):
        # [seq_len, batch_size] => [batch_size, seq_len]
        text = text.permute(1, 0) 
        # [batch_size, seq_len] => [batch_size, seq_len, embedding_size]
        embedded = self.embed(text) 
        # [batch_size, seq_len, embedding_size] => [batch_size, 1, seq_len, embedding_size]
        embedded = embedded.unsqueeze(1) 
        # [batch_size, 1, seq_len, embedding_size] => len(filter_sizes) of shape [batch_size, n_filters, seq_len-filter_size+1, 1]
        #                         conved.squeeze() => len(filter_sizes) of shape [batch_size, n_filters, seq_len-filter_size+1, 1]
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]  
        # len(filter_sizes) of shape [batch_size, n_filters, seq_len-filter_sizes+1] => len(filter_sizes) of shape [batch_size, n_filters, 1]
        #                                                           pooled.squeeze() => len(filter_sizes) of shape [batch_size, n_filters]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved] 
        # len(filter_sizes) of shape [batch_size, n_filters] => [batch_size, len(filter_sizes) * n_filters]
        pooled = torch.cat(pooled, dim = 1) 
        # [batch_size, len(filter_sizes) * n_filters] => [batch_size, output_size]
        out = self.linear(pooled)
        return out



# 定义模型训练函数
class EasyTrain:

    def train(self, model, iterator, optimizer, loss_fn):
        model.train()
        epoch_loss, epoch_acc, total_len = 0., 0., 0.
        for batch in iterator:
            preds = model(batch.text).squeeze() # [batch_size]
            loss = loss_fn(preds, batch.label) 
            acc = self.binary_accuracy(preds, batch.label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = len(batch.label)
            epoch_loss += loss.item() * batch_size
            epoch_acc += acc.item() * batch_size
            total_len += batch_size

        return epoch_loss/total_len, epoch_acc/total_len

    def evaluate(self, model, iterator, loss_fn):
        model.eval()
        epoch_loss, epoch_acc, total_len = 0., 0., 0.
        for batch in iterator:
            preds = model(batch.text).squeeze()
            loss = loss_fn(preds, batch.label)
            acc = self.binary_accuracy(preds, batch.label)

            batch_size = len(batch.label)
            epoch_loss += loss.item() * batch_size
            epoch_acc += acc.item() * batch_size
            total_len += batch_size
        model.train()

        return epoch_loss/total_len, epoch_acc/total_len

    def binary_accuracy(self, preds, y):
        rounded_preds = torch.round(torch.sigmoid(preds))
        correct = (rounded_preds == y).float()
        acc = correct.sum() / len(correct)
        return acc



def main():
    print("load data ...")
    #两个Field对象定义字段的处理方法（文本字段、标签字段）
    TEXT = data.Field(tokenize='spacy',
                    tokenizer_language='en_core_web_sm')
    LABEL = data.LabelField(dtype = torch.float)

    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
    train_data,val_data = train_data.split(split_ratio = 0.8, 
                                           strata_field = 'label',)
    print("build vocab ...")
    TEXT.build_vocab(train_data, 
                     max_size = 25000, 
                     vectors = "glove.6B.100d", # 导入预训练好的词向量
                     unk_init = torch.Tensor.normal_)
    LABEL.build_vocab(train_data)

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
                                                        (train_data, val_data, test_data), 
                                                        batch_size = BATCH_SIZE,
                                                        device = DEVICE)


    VOCAB_SIZE = len(TEXT.vocab)
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
    PRE_EMBEDED = TEXT.vocab.vectors

    # model = WordAVGModel(vocab_size = VOCAB_SIZE, 
    #                      embedding_size = EMBEDDING_SIZE, 
    #                      output_size = OUTPUT_SIZE, 
    #                      pad_idx = PAD_IDX)

    # HIDDEN_SIZE = 64
    # DROPOUT = 0.5
    # model = LSTMModel(vocab_size = VOCAB_SIZE, 
    #                  embedding_size = EMBEDDING_SIZE, 
    #                  pad_idx = PAD_IDX,
    #                  hidden_size = HIDDEN_SIZE,
    #                  output_size = OUTPUT_SIZE, 
    #                  dropout = DROPOUT
    #                  )

    # model = BiLSTMModel(vocab_size = VOCAB_SIZE, 
    #                  embedding_size = EMBEDDING_SIZE, 
    #                  pad_idx = PAD_IDX,
    #                  hidden_size = HIDDEN_SIZE,
    #                  output_size = OUTPUT_SIZE, 
    #                  rnn_layers = 2,
    #                  dropout = DROPOUT)

    
    # NUM_FILTERS = 64
    # FILTER_SIZES = [3]
    # DROPOUT = 0.5
    # model = CNNModel(vocab_size = VOCAB_SIZE, 
    #                 embedding_size = EMBEDDING_SIZE, 
    #                 n_filters = NUM_FILTERS, 
    #                 filter_sizes = FILTER_SIZES, 
    #                 output_size = OUTPUT_SIZE, 
    #                 dropout = DROPOUT, 
    #                 pad_idx = PAD_IDX)


    NUM_FILTERS = 64
    FILTER_SIZES = [3,4,5]
    DROPOUT = 0.5
    model = CNNsModel(vocab_size = VOCAB_SIZE, 
                    embedding_size = EMBEDDING_SIZE, 
                    n_filters = NUM_FILTERS, 
                    filter_sizes = FILTER_SIZES, 
                    output_size = OUTPUT_SIZE, 
                    dropout = DROPOUT, 
                    pad_idx = PAD_IDX)


    # 模型权重初始化
    model.embed.weight.data.copy_(PRE_EMBEDED)
    model.embed.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_SIZE)
    model.embed.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_SIZE)
    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()

    print("train ...")
    easytrain = EasyTrain()
    best_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = easytrain.train(model, train_iterator, optimizer, loss_fn)
        valid_loss, valid_acc = easytrain.evaluate(model, valid_iterator, loss_fn)
        if valid_loss < best_loss:
            best_loss = valid_loss
            # torch.save(model.state_dict(), "imdb-wordavg-model.pth")  
            # torch.save(model.state_dict(), "imdb-lstm-model.pth")  
            # torch.save(model.state_dict(), "imdb-bilstm-model.pth")  
            # torch.save(model.state_dict(), "imdb-cnn-model.pth") 
            torch.save(model.state_dict(), "imdb-cnns-model.pth")   
        print("Epoch", epoch, "Train Loss", train_loss, "Train Acc", train_acc)
        print("Epoch", epoch, "Valid Loss", valid_loss, "Valid Acc", valid_acc)

    print("test ...")
    model = model.load_state_dict(torch.load('imdb-cnns-model.pth'))
    test_loss, test_acc = easytrain.evaluate(model, test_iterator, loss_fn)
    print("-*"*5,"Test Loss", test_loss, "Test Acc", test_acc, "-*"*5)


if __name__ == '__main__':
    main()

