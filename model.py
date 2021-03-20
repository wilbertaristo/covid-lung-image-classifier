import numpy as np
import time
from collections import OrderedDict

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import models
from numpy import save, load
import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

# Define function to save checkpoint


def save_checkpoint(model, path):
    checkpoint = {'c_input': model.classifier.n_in,
                  'c_hidden': model.classifier.n_hidden,
                  'c_out': model.classifier.n_out,
                  'labelsdict': model.classifier.labelsdict,
                  'c_lr': model.classifier.lr,
                  'state_dict': model.state_dict(),
                  'c_state_dict': model.classifier.state_dict(),
                  'opti_state_dict': model.classifier.optimizer_state_dict,
                  'model_name': model.classifier.model_name,
                  'class_to_idx': model.classifier.class_to_idx
                  }
    torch.save(checkpoint, path)


class Net(nn.Module):
    def __init__(self, layers, with_dropout=True):
        super(Net, self).__init__()
        self.with_dropout = with_dropout
        self.conv = {}
        self.batchnorm = {}
        for i in range(layers):
            if i == 0:
                self.conv[i] = nn.Conv2d(1, 32, 3, 1).to('cuda')
                self.batchnorm[i] = nn.BatchNorm2D(32).to('cuda')
            else:
                self.conv[i] = nn.Conv2d(
                    32*(2**(i-1)), 32*(2**i), 3, 1).to('cuda')
                self.batchnorm[i] = nn.BatchNorm2D(32*(2**i)).to('cuda')
        if with_dropout:
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
        fc_lay = int(32 * (2 ** (layers-1)) * (((150-(layers * 2))/2)**2))
        self.fc1 = nn.Linear(fc_lay, 128)
        self.fc2 = nn.Linear(128, 32)
        self.classifier = nn.Linear(32, 2)

    def forward(self, x):
        for index in range(len(self.conv.values())):
            x = self.conv[index](x)
            x = F.relu(x)
            x = self.batchnorm[index](x)
        x = F.max_pool2d(x, 2)
        if self.with_dropout:
            x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        if self.with_dropout:
            x = self.dropout2(x)
        x = self.classifier(x)
        output = F.log_softmax(x, dim=1)
        return output


# Define function to load model
def load_model(path):
    cp = torch.load(path)

    # Import pre-trained NN model
    model = getattr(models, cp['model_name'])(pretrained=True)

    # Freeze parameters that we don't need to re-train
    for param in model.parameters():
        param.requires_grad = False

    # Make classifier
    model.classifier = NN_Classifier(input_size=cp['c_input'], output_size=cp['c_out'],
                                     hidden_layers=cp['c_hidden'])

    # Add model info
    model.classifier.n_in = cp['c_input']
    model.classifier.n_hidden = cp['c_hidden']
    model.classifier.n_out = cp['c_out']
    model.classifier.labelsdict = cp['labelsdict']
    model.classifier.lr = cp['c_lr']
    model.classifier.optimizer_state_dict = cp['opti_state_dict']
    model.classifier.model_name = cp['model_name']
    model.classifier.class_to_idx = cp['class_to_idx']
    model.load_state_dict(cp['state_dict'])

    return model


def test_model(model, testloader, device='cuda'):
    model.to(device)
    model.eval()
    accuracy = 0

    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)

        output = model.forward(images)

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    print('Testing Accuracy: {:.3f}'.format(accuracy/len(testloader)))


def init_weights(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def covid_model(n_hidden, n_epoch, labelsdict, lr, device, model_name, trainloader, validloader, train_data, layers):
    model = Net(layers=layers, with_dropout=False)

    model.to(device)
    model.apply(init_weights)
    print(model)

    # optimiser is using Adam for SGD and the learning rate is lr that will be controlled by a learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    loss_acc = []
    for epoch in range(1, n_epoch + 1):
        train(model, device, trainloader, optimizer, epoch)
        val_loss = evaluate(model, device, validloader, epoch)
        loss_acc.append(val_loss)
        scheduler.step()
    torch.save(model.state_dict(), "covid_model.pt")
    save('covid.npy', loss_acc)

    return model


def healthy_model(n_hidden, n_epoch, labelsdict, lr, device, model_name, trainloader, validloader, train_data, layers):
    model = Net(layers=layers)
    model.to(device)
    model.apply(init_weights)
    print(model)

    # optimiser is using Adam for SGD and the learning rate is lr that will be controlled by a learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    loss_acc = []
    for epoch in range(1, n_epoch + 1):
        train(model, device, trainloader, optimizer, epoch)
        val_loss = evaluate(model, device, validloader, epoch)
        loss_acc.append(val_loss)
        scheduler.step()
    torch.save(model.state_dict(), "healthy_model.pt")
    save('health.npy', loss_acc)

    return model


def evaluate(model, device, validloader, epoch):
    model.eval()
    val_loss = 0
    valcorrect = 0
    with torch.no_grad():
        for data2, target2 in validloader:
            data2, target2 = data2.to(device), target2.to(device)
            output2 = model(data2)
            val_loss = F.cross_entropy(
                output2, target2, reduction='sum').item()
            pred2 = output2.argmax(dim=1, keepdim=True)
            valcorrect += pred2.eq(target2.view_as(pred2)).sum().item()
    val_loss /= len(validloader.dataset)
    print('\nEpoch {} Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        epoch, val_loss, valcorrect, len(validloader.dataset), 100. * valcorrect / len(validloader.dataset)))
    return val_loss

# function to train the model


def train(model, device, train_loader, optimizer, epoch):
    print('model is training')
    # setting the model to training mode
    loss = 0
    correct_count = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        for j in range(len(target)):
            if pred[j] == target[j]:
                correct_count += 1
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        accuracy = correct_count / len(target)
        correct_count = 0
        if batch_idx % train_loader.batch_size == 0:
            print("Accuracy: {}".format(accuracy))
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
