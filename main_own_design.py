import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import *
import torch.optim as optim  # 导入torch.optim模块
import matplotlib.pyplot as plt
import numpy as np
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# define hyper-parameter
lr = 1e-4
batchsize = 1024
num_iteration=100
device='cuda:1'
model_save_path='./weights/own_design.pt'

# generate dataloader
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

dataset_train = torchvision.datasets.CIFAR10(root='./data',train=True,download=True, transform=transform_train)
dataloader_train = DataLoader(dataset_train,batch_size=batchsize,shuffle=True,num_workers=4)

dataset_test = torchvision.datasets.CIFAR10(root='./data',train=False,download=True, transform=transform_test)
dataloader_test = DataLoader(dataset_test,batch_size=batchsize,shuffle=True,num_workers=4)


# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# define model
model=MY_CNN().to(device)

criterion = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=lr)

best_acc=0.0
train_loss_lst=[]
test_loss_lst=[]
train_acc_lst=[]
test_acc_lst=[]
for epoch in range(num_iteration):
    print('epoch:%d'%epoch)
    model.train()
    train_loss=0.0
    correct=0
    seen=0
    for i, (img, labels) in enumerate(dataloader_train):
        img, labels=img.to(device), labels.to(device)
        opt.zero_grad()
        pred=model(img)
        loss=criterion(pred,labels)
        loss.backward()
        opt.step()

        train_loss+=loss.item()
        _,pred_class=pred.max(1)
        seen+=batchsize
        correct+=pred_class.eq(labels).sum().item()

        if seen/batchsize%500==0:
            print('[training] epoch:%2d, seen:%5d, loss:%.3f, accuracy:%.2f'
                  %(epoch,seen,train_loss/seen,100.0*correct/seen))

    print('[training] epoch:%2d, seen:%5d, loss:%.3f, accuracy:%.2f'
          % (epoch, seen, train_loss / seen, 100.0 * correct / seen))
    train_loss_lst.append(train_loss/ seen)
    train_acc_lst.append(100.0 * correct / seen)
    model.eval()
    test_loss = 0.0
    correct = 0
    seen = 0
    with torch.no_grad():
        for i, (img, labels) in enumerate(dataloader_test):
            img, labels=img.to(device), labels.to(device)
            pred=model(img)
            loss=criterion(pred,labels)

            test_loss+=loss.item()
            _,pred_class=pred.max(1)
            seen+=labels.size(0)
            correct+=pred_class.eq(labels).sum().item()
    test_loss_lst.append(test_loss/ seen)
    test_acc_lst.append(100.0 * correct / seen)
    print('[testing] epoch:%2d, seen:%5d, loss:%.3f, accuracy:%.2f'
          %(epoch,seen,test_loss/seen,100.0*correct/seen))

    acc=100.0*correct/seen
    if acc>best_acc:
        ckpt=model.state_dict()
        torch.save(ckpt,model_save_path)
        best_acc=acc
        print('Model has been updating!')
plt.figure(1,figsize=(17.5, 5))
plt.subplot(1,2,1)
plt.plot(np.arange(num_iteration),train_loss_lst,label='Train loss')
plt.plot(np.arange(num_iteration),test_loss_lst,label='Test loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1,2,2)
plt.plot(np.arange(num_iteration),train_acc_lst,label='Train accuracy')
plt.plot(np.arange(num_iteration),test_acc_lst,label='Test accuracy')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('own_design_1e-4.png')

