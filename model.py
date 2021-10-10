import torch.nn as nn
import torch.nn.functional as F
import torchvision

class MY_CNN(nn.Module):
    def __init__(self):
        super(MY_CNN,self).__init__()
        self.conv1=nn.Conv2d(3,16,5)
        self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(16,32,5)
        self.fc1=nn.Linear(32*5*5,512)
        self.fc2=nn.Linear(512,256)
        self.fc3=nn.Linear(256,10)

    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=x.view(x.size(0), -1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        return x

def alexnet():
    return  torchvision.models.alexnet()

def vgg16():
    return torchvision.models.vgg16()

def resnet():
    return torchvision.models.resnet50()