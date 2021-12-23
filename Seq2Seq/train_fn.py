
import numpy as np
import torch
from torch import from_numpy

def train(model, data, num_epochs, loss_fn, optimizer, valid_data, device):
    for epoch in range(num_epochs):
        model.train()
        total_num_words = total_loss = 0.
        for it, (en_input, en_len, de_input, de_len) in enumerate(data):#（英文batch，英文长度，中文batch，中文长度）
            en_input,en_len = from_numpy(en_input).to(device).long(), from_numpy(en_len).to(device).long()
            #前n-1个单词作为输入，后n-1个单词作为输出，因为输入的前一个单词要预测后一个单词
            de_input, output = from_numpy(de_input[:, :-1]).to(device).long(),from_numpy(de_input[:, 1:]).to(device).long()
            de_len = from_numpy(de_len-1).to(device).long() # 输入输出的长度都减1
            de_len[de_len<=0] = 1
            pred, attn = model(en_input, en_len, de_input, de_len) # 返回的是类PlainSeq2Seq里forward函数的两个返回值
            out_mask = torch.arange(de_len.max().item(), device=device)[None, :] < de_len[:, None]
            out_mask = out_mask.float()
            # out_mask就是LanguageModelCriterion的传入参数mask。
            loss = loss_fn(pred, output, out_mask)
            num_words = torch.sum(de_len).item() #一个batch里多少个单词
            total_loss += loss.item() * num_words
            #总损失，loss计算的是均值损失，每个单词都是都有损失，所以乘以单词数
            total_num_words += num_words
            # 更新模型
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.) #为了防止梯度过大，设置梯度的阈值
            optimizer.step()
            
            if it % 200 == 0:
                print("Epoch: %d, iteration: %d, loss: %.4f"%(epoch, it, loss.item()))

        print("\nEpoch: %d, Training loss: %.4f \n"%(epoch, total_loss/total_num_words))
        if epoch % 10 == 0:
            evaluate(model, data = valid_data, loss_fn = loss_fn, device = device) # 评估模型

    return model

def evaluate(model, data, loss_fn, device):
    model.eval()
    total_num_words = total_loss = 0.
    with torch.no_grad():
        for it, (en_input, en_len, de_input, de_len) in enumerate(data):
            en_input = from_numpy(en_input).long().to(device)
            en_len = from_numpy(en_len).long().to(device)
            trans_input = from_numpy(de_input[:,:-1]).long().to(device)
            output = from_numpy(de_input[:, 1:]).long().to(device)
            de_len = from_numpy(de_len-1).long().to(device)
            de_len[de_len <= 0] = 1

            pred, attn = model(en_input, en_len, trans_input, de_len)

            out_mask = torch.arange(de_len.max().item(), device=device)[None, :] < de_len[:, None]
            out_mask = out_mask.float()
            loss = loss_fn(pred, output, out_mask)

            num_words = torch.sum(de_len).item()
            total_loss += loss.item() * num_words
            total_num_words += num_words

    print("evaluation loss %.4f"% (total_loss/total_num_words))

def translate_dev(model, device, dev_en, inv_en_dict, dev_cn, inv_cn_dict,cn_dict,i):
    en_sent = " ".join([inv_en_dict[w] for w in dev_en[i]])
    print(en_sent)
    cn_sent = " ".join([inv_cn_dict[w] for w in dev_cn[i]])
    print("".join(cn_sent))
    mb_x = torch.from_numpy(np.array(dev_en[i]).reshape(1, -1)).long().to(device)
    #把句子升维，并转换成tensor
    mb_x_len = torch.from_numpy(np.array([len(dev_en[i])])).long().to(device)
    #取出句子长度，并转换成tensor
    bos = torch.Tensor([[cn_dict["BOS"]]]).long().to(device)
    translation, attn = model.translate(mb_x, mb_x_len, bos)
    #这里传入bos作为首个单词的输入
    #translation=tensor([[ 8,  6, 11, 25, 22, 57, 10,  5,  6,  4]])
    translation = [inv_cn_dict[i] for i in translation.data.cpu().numpy().reshape(-1)]
    trans = []
    for word in translation:
        if word != "EOS": # 把数值变成单词形式
            trans.append(word) #
        else:
            break
    print("".join(trans))   
