# -*- coding: utf-8 -*-
"""
Created on Tue May 18 23:34:59 2021

@author: Peter
"""
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 10:21:22 2021

@author: Peter
"""

import numpy as np
from torch import nn
from einops.layers.torch import Rearrange

import torch

import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
import os

transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomGrayscale(p=0.5),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

transform1 = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=50,
                                          shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform1)

testloader = torch.utils.data.DataLoader(testset, batch_size=50,
                                         shuffle=True)
# =============================================================================

# =============================================================================
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MixerBlock(nn.Module):

    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout=0.3):
        super().__init__()

        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b d n -> b n d')
        )

        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )

    def forward(self, x):
        x = x + self.token_mix(x)

        x = x + self.channel_mix(x)

        return x


class MLPMixer(nn.Module):

    def __init__(self, in_channels, dim, num_classes, patch_size, image_size, depth, token_dim, channel_dim):
        super().__init__()

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.num_patch = (image_size // patch_size) ** 2
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, dim, patch_size, patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )

        self.mixer_blocks = nn.ModuleList([])

        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(dim, self.num_patch, token_dim, channel_dim))

        self.layer_norm = nn.LayerNorm(dim)

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):

        x = self.to_patch_embedding(x)

        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)

        x = self.layer_norm(x)

        x = x.mean(dim=1)

        return self.mlp_head(x)


# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = MLPMixer(in_channels=3, image_size=32, patch_size=4, num_classes=10,
                 dim=768, depth=20, token_dim=384, channel_dim=3072)

model = model.to(device)

path = '20210524pm.pth'

# loss
CE_loss = nn.CrossEntropyLoss()

# 优化器

optimizer = optim.Adam(model.parameters(), lr=0.0001,weight_decay=1e-5,betas=(0.9,0.999))

def train():
    # 训练状态
    start_time = time.time()
    if os.path.exists(path) is True:
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # initepoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print("load success!")
    model.train()

    loss_all=0

    for i, data in enumerate(trainloader):
        # 获得一个批次的标签和数据
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # 获得模型的结果
        out = model(inputs)

        loss = CE_loss(out, labels)
        loss_all+=loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
                }, path)
    print('loss:',loss_all)

    end_time = time.time()

    print('time cost for train:', end_time - start_time)


acc_list=[]
best='best.pth'
def test():
    time_start=time.time()
    model.eval()
    #计算训练的准确率
    correct=0
    for i, data in enumerate(trainloader):
        # 获得一个批次的标签和数据
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # 获得模型的结果
        out = model(inputs)

        _, predicted = torch.max(out, 1)

        correct += (predicted == labels).sum()

    print("Test acc for train:{0}".format(correct.item() / len(trainset)))
    f=open('acc_train.txt','a')
    f.write(str(correct.item()/len(trainset)))
    f.write('\n')
    f.close()


    # 计算测试的准确率

    correct = 0
    for i, data in enumerate(testloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        out = model(inputs)

        _, predicted = torch.max(out, 1)

        correct += (predicted == labels).sum()

    accuracy=correct.item() / len(testset)
    acc_list.append(accuracy)

    if max(acc_list)==accuracy:
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    #'loss': loss
                    }, best)
        print('best result!')

    print("Test acc for test:{0}".format(accuracy))
    f=open('acc_test.txt','a')
    f.write(str(correct.item() / len(testset)))
    f.write('\n')
    f.close()

    time_end=time.time()
    print('time cost for test:',time_end-time_start)



for epoch in range(100):
    print("")
    print(epoch)
    train()
    test()



