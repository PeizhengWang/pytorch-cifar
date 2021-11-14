import torch.nn as nn
import torch.nn.functional as F
import torchvision

class MY_CNN(nn.Module):
    def __init__(self):
        super(MY_CNN,self).__init__()
        self.conv1=nn.Conv2d(3,16,3,1,padding=1)
        self.conv2=nn.Conv2d(16,32,3,1,padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, 1, padding=1)
        self.conv5 = nn.Conv2d(64, 128, 3, 1, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.dropout1 = nn.Dropout(0.5)
        self.fc1=nn.Linear(128*4*4,512)
        self.fc2=nn.Linear(512,10)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 128)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5,self).__init__()
        self.conv1=nn.Conv2d(3,6,5,1)
        self.conv2=nn.Conv2d(6,16,5,1)
        self.fc1=nn.Linear(16*5*5,512)
        self.fc2=nn.Linear(512,84)
        self.fc3=nn.Linear(84,10)

    def forward(self,x):
        x=F.max_pool2d(F.relu(self.conv1(x)),2,2)
        x=F.max_pool2d(F.relu(self.conv2(x)),2,2)
        x=x.view(-1, 16*5*5)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.softmax(self.fc3(x))
        return x


def alexnet():
    return  torchvision.models.alexnet()

def vgg16():
    return torchvision.models.vgg16()

def resnet():
    return torchvision.models.resnet50()