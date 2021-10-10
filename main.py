import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import *
import torch.optim as optim  # 导入torch.potim模块


import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# define hyper-parameter
lr = 0.001
batchsize = 8
num_iteration=100
device='cuda'
model_save_path='./weights/MY_CNN.pt'
# generate dataloader
transform=transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
dataset_train = torchvision.datasets.CIFAR10(root='./data',train=True,download=True, transform=transform)
dataloader_train = DataLoader(dataset_train,batch_size=batchsize,shuffle=True,num_workers=4)

dataset_test = torchvision.datasets.CIFAR10(root='./data',train=False,download=True, transform=transform)
dataloader_test = DataLoader(dataset_train,batch_size=batchsize,shuffle=True,num_workers=4)


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# define model
model=LeNet5()
model.to(device)
criterion = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=lr)

best_acc=0.0
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
            seen=0
            train_loss=0
            correct=0
    print('[training] epoch:%2d, seen:%5d, loss:%.3f, accuracy:%.2f'
          % (epoch, seen, train_loss / seen, 100.0 * correct / seen))

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


    print('[testing] epoch:%2d, seen:%5d, loss:%.3f, accuracy:%.2f'
          %(epoch,seen,test_loss/seen,100.0*correct/seen))

    acc=100.0*correct/seen
    if acc>best_acc:
        ckpt=model.state_dict()
        torch.save(ckpt,model_save_path)
        best_acc=acc
        print('Model has been updating!')