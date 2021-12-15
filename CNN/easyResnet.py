#coding:utf-8
#date:2021-12-15

import numpy as np
import torch,torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import time
import os
import copy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, device, dataloader, loss_fn, optimizer, num_epochs = 5):
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.
    for epoch in range(num_epochs):
        for phase in ["train", "val"]:
            model.train()
            if phase == "val":model.eval()
            
            batch_loss, batch_corrects = 0.,0.
            for inputs, labels in dataloader[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.autograd.set_grad_enabled(phase=="train"):
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
                if phase == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()  
                    
                _, preds = torch.max(outputs, 1)
                batch_loss += loss.item() * inputs.size(0)
                batch_corrects += torch.sum(preds.view(-1) == labels.view(-1)).item()
            
            epoch_loss = batch_loss / len(dataloader[phase].dataset)
            epoch_acc = batch_corrects / len(dataloader[phase].dataset)
            print("Epoch {}: {}-Loss: {} Acc: {}".format(epoch, phase, epoch_loss, epoch_acc))
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == "val":
                val_acc_history.append(epoch_acc)
    
    time_elapsed = time.time() - since
    print("Training compete in {}m {}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best val Acc: {}".format(best_acc))
    
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def initialize_model(model_name, num_classes, use_pretrained = True):
    """if use_pretrained = True, 
    """
    if model_name == "resnet":
        model = models.resnet18(pretrained = use_pretrained)
        if use_pretrained:
            for param in model.parameters():
                param.requires_grad = False


        num_feats = model.fc.in_features
        model.fc = nn.Linear(num_feats, num_classes) # out_features=1000 改为 num_classes=2
        input_size = 224 # resnet18网络输入图片维度是224，resnet34，50，101，152也是

    return model, input_size

def update_params(model, use_pretrained = True):
    if use_pretrained:
        params_to_update = []
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    return params_to_update

def data_load(data_path, input_size, batch_size):
    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("Initializing Datasets and Dataloaders...")
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_path, x), data_transforms[x]) 
                      for x in ['train', 'val']}

    dataloaders_dict = {x: DataLoader(image_datasets[x], 
                                      batch_size = batch_size, 
                                      shuffle = True, 
                                      num_workers = 4) 
                        for x in ['train', 'val']}

    return dataloaders_dict


def main():
    data_dir = "./hymenoptera_data"
    # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
    model_name = "resnet"
    num_classes = 2
    batch_size = 32
    num_epochs = 2
    use_pretrained = True #只更新修改的层
    save_model = True

    model, input_size = initialize_model(model_name, 
                                        num_classes, 
                                        use_pretrained = use_pretrained)
    
    dataloaders_dict = data_load(data_path = data_dir, 
                                input_size = input_size,
                                batch_size = batch_size)

    print("Params to learn:")
    params_to_update = update_params(model, use_pretrained = use_pretrained)

    model = model.to(DEVICE)
    optimizer = SGD(params_to_update, lr=0.001, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    # Train and evaluate
    model, history = train_model(model, 
                               device = DEVICE,
                               dataloader = dataloaders_dict, 
                               loss_fn = loss_fn, 
                               optimizer = optimizer, 
                               num_epochs = num_epochs)

    if (save_model):
        torch.save(model.state_dict(),"./easyresnet.th")


if __name__ == '__main__':
    main()
    print(" ----------------- finished -----------------")
