#coding=utf-8

from __future__ import print_function,division
from vgg16 import vgg16
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

batch_size = 64
learning_rate = 0.0002
epoch = 10


#用于图片增强
train_transforms = transforms.Compose([
    transforms.Resize(256), #转换成256大小·
    transforms.RandomResizedCrop(224), #随机裁剪
    transforms.RandomHorizontalFlip(), #随机翻转
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])





#加载数据机
train_dir = './data/traindata'
train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
 
val_dir = './data/valdata'
val_datasets = datasets.ImageFolder(val_dir, transform=val_transforms)
val_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size, shuffle=True)
 
#==============================训练过程=========================================

model = vgg16(num_classes=3)
#检测网络是否可用
if torch.cuda.is_available():
    torch.cuda.set_device(3)
    model.cuda() 
# params = [{'params': md.parameters()} for md in model.children()
#           if md in [model.classifier]]



#加载预训练模型

net = torch.load("./model/vgg16-397923af.pth")


pre_dict = net


print(pre_dict.keys(),"predict")
model_dict = model.state_dict()

print(model_dict.keys(), "model_dict")



update_dict = {k:v for k,v in pre_dict.items() if k in model_dict}

print(update_dict.keys(),"update dict info")

model_dict.update(update_dict)

model.load_state_dict(model_dict)




optimizer = optim.Adam(model.parameters(), lr=learning_rate) # 设置优化器
loss_func = nn.CrossEntropyLoss() #设置损失函数
 
Loss_list = []
Accuracy_list = []
 
for epoch in range(5):
    print('epoch {}'.format(epoch + 1))
    # training-----------------------------
    train_loss = 0.
    train_acc = 0.
    for batch_x, batch_y in tqdm(train_dataloader):
        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
 
        out = model(batch_x)
 
        loss = loss_func(out, batch_y)
 
        train_loss += loss.item()
        pred = torch.max(out, 1)[1]
 
        train_correct = (pred == batch_y).sum()
        train_acc += train_correct.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(
        train_datasets)), train_acc / (len(train_datasets))))
 
    # evaluation--------------------------------
    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    for batch_x, batch_y in tqdm(val_dataloader):
        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

        out = model(batch_x)
        loss = loss_func(out, batch_y)
        eval_loss += loss.item()
        pred = torch.max(out, 1)[1]
        num_correct = (pred == batch_y).sum()
        eval_acc += num_correct.item()
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        val_datasets)), eval_acc / (len(val_datasets))))
 
    Loss_list.append(eval_loss / (len(val_datasets)))
    Accuracy_list.append(100 * eval_acc / (len(val_datasets)))
#模型保存
torch.save(model, './model/model.pth')






