# Pytorch-Cifar
A pytorch implementation of several neural networks trained on the CIFAR-10 dataset. 

# Dataset
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

Here are the classes in the dataset, as well as 10 random images from each:

![cifar](/asset/cifar.png)

The classes are completely mutually exclusive. There is no overlap between automobiles and trucks. "Automobile" includes sedans, SUVs, things of that sort. "Truck" includes only big trucks. Neither includes pickup trucks.

# Experiment
## Settings
- batch_size: 1024
- device: Tesla P100
- Environment
	- Python	=	3.7.11
	- torch		=	1.9.0
	- numpy	=	1.21.2
	- torchvision = 0.10.0
## Model
We trained two models, one is LeNet5, and the other is a model designed by ourselves. The architecture of model is as follows:
```python
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
```
We replaced the 5*5 convolution kernel with a 3*3 convolution kernel and used more convolution layers.
## Result
|Network|Learning rate|Epoch|Accuracy|
|-|-|-|-|
|LeNet5|1e-2|20|10.00|
|LeNet5|1e-3|40|75.18|
|Own Network|1e-2|20|67.88|
|Own Network|1e-3|40|99.54|
-----------------
We will show learning curve as follows:

LeNet5 with lr=1e-2:
![LeNe5e-2](/asset/lenet5_1e-2.png)
The model failed to converge due to the excessive learning rate.

LeNet5 with lr=1e-3:
![LeNe5e-3](/asset/lenet5_1e-3.png)

Own Network with lr=1e-2:
![Own Networke-2](/asset/own_design_1e-2.png)

Own Network with lr=1e-3:
![Own Networke-3](/asset/own_design_1e-3.png)

# Reference
- https://www.cs.toronto.edu/~kriz/cifar.html
- https://github.com/kuangliu/pytorch-cifar
