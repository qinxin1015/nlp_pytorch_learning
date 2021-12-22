
import torch

def train(model, data, num_epochs, loss_fn, optimizer, valid_data, device):
    for epoch in range(num_epochs):
        total_num_words = total_loss = 0.
        model.train()
        for it, (mb_x, mb_x_lengths, mb_y, mb_y_lengths) in enumerate(data):
            mb_x = torch.from_numpy(mb_x).long().to(device)
            mb_x_lengths = torch.from_numpy(mb_x_lengths).long().to(device)
            mb_input = torch.from_numpy(mb_y[:,:-1]).long().to(device)
            mb_out = torch.from_numpy(mb_y[:, 1:]).long().to(device)
            mb_y_lengths = torch.from_numpy(mb_y_lengths-1).long().to(device)
            mb_y_lengths[mb_y_lengths <= 0] = 1

            mb_pred, attn = model(mb_x, mb_x_lengths, mb_input, mb_y_lengths)

            mb_out_mask = torch.arange(mb_y_lengths.max().item(), device=device)[None, :] < mb_y_lengths[:, None]
            mb_out_mask = mb_out_mask.float()
            loss = loss_fn(mb_pred, mb_out, mb_out_mask)

            num_words = torch.sum(mb_y_lengths).item()
            total_loss += loss.item() * num_words
            total_num_words += num_words

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()

            if it % 100 == 0:
                print("epoch", epoch, "iteration", it, "loss", loss.item())
        print("epoch", epoch, "training loss", total_loss/total_num_words)
        if epoch % 5 == 0:
            print("evaluating on dev...")
            evaluate(model, valid_data, loss_fn = loss_fn, device = device)

def evaluate(model, data, loss_fn, device):
    model.eval()
    total_num_words = total_loss = 0.
    with torch.no_grad():
        for it, (mb_x, mb_x_lengths, mb_y, mb_y_lengths) in enumerate(data):
            mb_x = torch.from_numpy(mb_x).long().to(device)
            mb_x_lengths = torch.from_numpy(mb_x_lengths).long().to(device)
            mb_input = torch.from_numpy(mb_y[:,:-1]).long().to(device)
            mb_out = torch.from_numpy(mb_y[:, 1:]).long().to(device)
            mb_y_lengths = torch.from_numpy(mb_y_lengths-1).long().to(device)
            mb_y_lengths[mb_y_lengths <= 0] = 1

            mb_pred, attn = model(mb_x, mb_x_lengths, mb_input, mb_y_lengths)

            mb_out_mask = torch.arange(mb_y_lengths.max().item(), device=device)[None, :] < mb_y_lengths[:, None]
            mb_out_mask = mb_out_mask.float()
            loss = loss_fn(mb_pred, mb_out, mb_out_mask)

            num_words = torch.sum(mb_y_lengths).item()
            total_loss += loss.item() * num_words
            total_num_words += num_words

    print("evaluation loss", total_loss/total_num_words)

def translate_dev(i):
    model.eval()
    en_sent = " ".join([inv_en_dict[word] for word in dev_en[i]])
    print(en_sent)
    print(" ".join([inv_cn_dict[word] for word in dev_cn[i]]))

    sent = nltk.word_tokenize(en_sent.lower())
    bos = torch.Tensor([[cn_dict["BOS"]]]).long().to(device)
    mb_x = torch.Tensor([[en_dict.get(w, 0) for w in sent]]).long().to(device)
    mb_x_len = torch.Tensor([len(sent)]).long().to(device)
    
    translation, attention = model.translate(mb_x, mb_x_len, bos)
    translation = [inv_cn_dict[i] for i in translation.data.cpu().numpy().reshape(-1)]

    trans = []
    for word in translation:
        if word != "EOS":
            trans.append(word)
        else:
            break
    print(" ".join(translation))
