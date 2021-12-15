#coding:utf-8
#date:2021-12-15

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class easyNet(nn.Module):
    """MNIST shape [1,28,28]
    input -> [conv -> max_pool] * 2 -> reshape(view) -> fc -> fc -> output

    """
    def __init__(self):
        super(easyNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))     # [1,28,28] => [20,24,24]
        x = F.max_pool2d(x, 2, 2)     #           => [20,12,12]
        x = F.relu(self.conv2(x))     #           => [50, 8, 8]
        x = F.max_pool2d(x, 2, 2)     #           => [50, 4, 4]
        x = x.view(-1, 4*4*50)        #           => [1, 50*4*4]
        x = F.relu(self.fc1(x))       #           => [1, 500]
        x = self.fc2(x)               #           => [1, 10]
        return F.log_softmax(x, dim=1)#           => [1, 10]



def train(model, device, train_loader, optimizer, epoch, log_interval = 100):
    model.train()
    for idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = F.nll_loss(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if idx % log_interval == 0:
            print("Train Epoch: {} [{}/{} ({:4f}%)]\tLoss: {:.6f}".format(
                epoch, idx * len(data), len(train_loader.dataset), 
                100. * idx / len(train_loader), loss.item()
            ))
                     
def test(model, device, test_loader):
    model.eval()
    test_loss,correct = 0., 0.
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.6f}, Accuracy: {}/{} ({:.4f}%)\n'.format(test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))



def main():
    NUM_EPOCHS = 2
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    MOMENTUM = 0.9
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
    save_model = True

    train_data = datasets.MNIST('./mnist_data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    test_data = datasets.MNIST('./mnist_data', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))

    train_loader = DataLoader(train_data,
                              batch_size = BATCH_SIZE, 
                              shuffle = True, 
                              num_workers = 1, 
                              pin_memory = True)
    test_loader = DataLoader(test_data,
                              batch_size = BATCH_SIZE, 
                              shuffle = False, 
                              num_workers = 1, 
                              pin_memory = True)

    model = easyNet().to(DEVICE)
    optimizer = SGD(model.parameters(), 
                  lr = LEARNING_RATE, 
                  momentum = MOMENTUM)

    for epoch in range(NUM_EPOCHS):
        train(model, DEVICE, train_loader, optimizer, epoch, log_interval = 100)
        test(model, DEVICE, test_loader)
        
    if (save_model):
        torch.save(model.state_dict(),"./mnist_easycnn.th")


if __name__ == '__main__':
    main()