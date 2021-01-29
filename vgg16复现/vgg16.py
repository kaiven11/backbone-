#coding=utf-8
import torch
from torch import nn
class vgg16(nn.Module):
    def __init__(self,num_classes):
        super(vgg16,self).__init__()
        self.num_classes = num_classes
   
        #输入图片大小 224*224*3
        
        net = []

        #block 1
        net.append(nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1,stride=1))
        net.append(nn.ReLU())
        net.append(nn.MaxPool2d(kernel_size=2,stride=2)) #?

        #block 2
        net.append(nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels=128,out_channels=128,stride=1,padding=1,kernel_size=3))
        net.append(nn.ReLU())
        #block 3

        net.append(nn.MaxPool2d(kernel_size=2,stride=2))
       
        net.append(nn.Conv2d(in_channels=128,out_channels=256,stride=1,padding=1,kernel_size=3))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels=256,out_channels=256,stride=1,padding=1,kernel_size=3))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels=256,out_channels=256,stride=1,padding=1,kernel_size=3))
        net.append(nn.ReLU())


        #block 4

        net.append(nn.MaxPool2d(kernel_size=2,stride=2))#?
        net.append(nn.Conv2d(in_channels=256,out_channels=512,stride=1,padding=1,kernel_size=3))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels=512,out_channels=512,stride=1,padding=1,kernel_size=3))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels=512,out_channels=512,stride=1,padding=1,kernel_size=3))
        net.append(nn.ReLU())

        #block 5

        net.append(nn.MaxPool2d(stride=2,kernel_size=2))

        net.append(nn.Conv2d(in_channels=512,out_channels=512,stride=1,padding=1,kernel_size=3))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels=512,out_channels=512,stride=1,padding=1,kernel_size=3))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels=512,out_channels=512,stride=1,padding=1,kernel_size=3))
        net.append(nn.ReLU())

        #block 6

        net.append(nn.MaxPool2d(stride=2,kernel_size=2))
        self.extract_feature = nn.Sequential(*net)
        



        classifier = []
        classifier.append(nn.Linear(in_features=512*7*7, out_features=4096))
        classifier.append(nn.ReLU())
        classifier.append(nn.Dropout(p=0.5))
        classifier.append(nn.Linear(in_features=4096, out_features=4096))
        classifier.append(nn.ReLU())
        classifier.append(nn.Dropout(p=0.5))
        classifier.append(nn.Linear(in_features=4096, out_features=self.num_classes))


        self.classifier = nn.Sequential(*classifier)

    def forward(self, x):
        feature = self.extract_feature(x)
        feature = feature.view(x.size(0), -1) # 转换为一纬度向量
        classify_result = self.classifier(feature)
        return classify_result








if __name__ == "__main__":
    x = torch.randn(size=(8, 3, 224, 224))
    vgg = vgg16(num_classes=1000)
    out = vgg(x)
    print(out.size())


    
