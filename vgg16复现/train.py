#coding=utf-8

from __future__ import print_function,division
from vgg16 import vgg16
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

batch_size = 8
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

model = vgg16(num_classes=2)
#检测网络是否可用
if torch.cuda.is_available():
    torch.cuda.set_device(3)
    model.cuda() 
params = [{'params': md.parameters()} for md in model.children()
          if md in [model.classifier]]
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_func = nn.CrossEntropyLoss()
 
Loss_list = []
Accuracy_list = []
 
for epoch in range(100):
    print('epoch {}'.format(epoch + 1))
    # training-----------------------------
    train_loss = 0.
    train_acc = 0.
    for batch_x, batch_y in train_dataloader:
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
    for batch_x, batch_y in val_dataloader:
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
#loss显示
x1 = range(0, 100)
x2 = range(0, 100)
y1 = Accuracy_list
y2 = Loss_list
plt.subplot(2, 1, 1)
plt.plot(x1, y1, 'o-')
plt.title('Test accuracy vs. epoches')
plt.ylabel('Test accuracy')
plt.subplot(2, 1, 2)
plt.plot(x2, y2, '.-')
plt.xlabel('Test loss vs. epoches')
plt.ylabel('Test loss')
plt.savefig("./result.jpg")





