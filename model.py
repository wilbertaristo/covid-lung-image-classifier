import numpy as np
import time
from collections import OrderedDict
import datetime
import pytz

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

# Define function to save checkpoint


class Net(nn.Module):
    def __init__(self, layers, model_name, with_dropout=True):
        super(Net, self).__init__()
        self.with_dropout = with_dropout
        self.conv = {}
        self.batchnorm = {}
        self.model_name = model_name
        for i in range(layers):
            if i == 0:
                self.conv[i] = nn.Conv2d(1, 16, 3, 1).to('cuda')
                self.batchnorm[i] = nn.BatchNorm2d(16).to('cuda')
            else:
                self.conv[i] = nn.Conv2d(
                    16*(2**(i-1)), 16*(2**i), 3, 1).to('cuda')
                self.batchnorm[i] = nn.BatchNorm2d(16*(2**i)).to('cuda')
        if with_dropout:
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
        fc_lay = int(16 * (2 ** (layers-1)) * (((150-(layers * 2))/2)**2))
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
    return cp


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
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def run_model(n_hidden, n_epoch, labelsdict, lr, device, model_name, trainloader, validloader, train_data, layers, covid=False):
    model = Net(layers=layers, model_name=model_name, with_dropout=(not covid))
    model.to(device)
    model.apply(init_weights)
    print(model)

    # optimiser is using Adam for SGD and the learning rate is lr that will be controlled by a learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr)
    training_loss = []
    validation_loss = []
    pending_loss = []
    previous_loss = 10 ** 99
    current_loss = 10 ** 99
    consecutive_increase = 0
    save_model = True
    min_epoch = 3
    for epoch in range(1, n_epoch + 1):
        train_loss, total_count, model = train(
            model, device, trainloader, optimizer, epoch, save_model)
        utc_now = pytz.utc.localize(datetime.datetime.utcnow())
        pst_now = utc_now.astimezone(pytz.timezone('Asia/Singapore'))
        time = pst_now.strftime("%Y-%m-%d %H:%M:%S")
        print('\nEpoch {} Time: {} Training set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(epoch,
                                                                                                           time, train_loss, total_count, len(trainloader.dataset), 100. * total_count / len(trainloader.dataset)))
        training_loss.append(train_loss)
        val_loss = evaluate(model, device, validloader, epoch)
        current_loss = val_loss

        # ==== Implement Early Stoppage if Overfitting ====
        if epoch > min_epoch:
            if current_loss < previous_loss:
                previous_loss = current_loss
                consecutive_increase = 0
                validation_loss += pending_loss
                validation_loss.append(current_loss)
                pending_loss = []
                save_model = True
            else:
                consecutive_increase += 1
                pending_loss.append(current_loss)
                save_model = False

            if save_model:
                print("Saving Model...")
                torch.save(model, "{}_model.pt".format(model.model_name))

            if consecutive_increase == 3:
                break
        else:
            previous_loss = current_loss
            validation_loss.append(current_loss)
            print("Saving Model...")
            torch.save(model, "{}_model.pt".format(model.model_name))
        # ==================================================

    save('{}.npy'.format('covid' if covid else 'health'), validation_loss)
    save('{}_train_loss.npy'.format('covid' if covid else 'health'),
         training_loss[:len(validation_loss)])

    return model


def evaluate(model, device, validloader, epoch):
    model.eval()
    val_loss = 0
    valcorrect = 0
    with torch.no_grad():
        for data2, target2 in validloader:
            data2, target2 = data2.to(device), target2.to(device)
            output2 = model(data2)
            val_loss += F.cross_entropy(
                output2, target2, reduction='sum').item()
            pred2 = output2.argmax(dim=1, keepdim=True)
            valcorrect += pred2.eq(target2.view_as(pred2)).sum().item()
    val_loss /= len(validloader.dataset)
    utc_now = pytz.utc.localize(datetime.datetime.utcnow())
    pst_now = utc_now.astimezone(pytz.timezone('Asia/Singapore'))
    time = pst_now.strftime("%Y-%m-%d %H:%M:%S")
    print('\nEpoch {} Time {} Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        epoch, time, val_loss, valcorrect, len(validloader.dataset), 100. * valcorrect / len(validloader.dataset)))
    return val_loss


def evaluate_without_training(model, device, validloader):
    model.eval()
    val_loss = 0
    valcorrect = 0
    with torch.no_grad():
        for data2, target2 in validloader:
            data2, target2 = data2.to(device), target2.to(device)
            output2 = model(data2)
            val_loss += F.cross_entropy(
                output2, target2, reduction='sum').item()
            pred2 = output2.argmax(dim=1, keepdim=True)
            valcorrect += pred2.eq(target2.view_as(pred2)).sum().item()
    val_loss /= len(validloader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, valcorrect, len(validloader.dataset), 100. * valcorrect / len(validloader.dataset)))

# function to train the model


def train(model, device, train_loader, optimizer, epoch, save_model):
    print('model is training')
    # setting the model to training mode
    loss = 0
    correct_count = 0
    total_loss = 0
    total_count = 0
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
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        accuracy = correct_count / len(target)
        total_count += correct_count
        correct_count = 0
        if batch_idx % train_loader.batch_size == 0:
            print("Accuracy: {}".format(accuracy))
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    lossout = total_loss / len(train_loader)
    return lossout, total_count, model
