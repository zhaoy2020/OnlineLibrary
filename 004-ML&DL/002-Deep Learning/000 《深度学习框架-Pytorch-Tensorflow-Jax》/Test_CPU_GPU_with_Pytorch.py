# 数据准备

import torch 
import torch.nn as nn 
# import torch.nn.functional as F 
import torch.utils.data as data

import torchvision

import time 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dbs = './Pytorch_datasets/'
train_dataset = torchvision.datasets.MNIST(root=dbs, 
                                           train=True, 
                                           download=True, 
                                           transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                                                                                    #  torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                                                                     ]
                                                                                     )
                                                                                     )
test_dataset = torchvision.datasets.MNIST(root=dbs, 
                                           train=False, 
                                           download=True, 
                                           transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                                                                                    #  torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                                                                     ]
                                                                                     )
                                                                                     )
# 迭代型数据方式
train_iter = data.DataLoader(dataset=train_dataset, 
                             batch_size=128, 
                             shuffle=True)
# test_iter = data.DataLoader(dataset=test_dataset) # test不需要batch训练

# 网络结构
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Flatten(),
                                     nn.Linear(28*28, 256), nn.ReLU(),
                                     nn.Linear(256, 10), nn.Softmax())
    def forward(self, X):
        return self.network(X)
    
# 训练过程封装
def train_steps(epochs, train_dataset, train_iter, test_dataset, net, loss_fn, opt, device):
    '''
    参数记录
    epochs = epochs                         # epoch
    train_dataset = train_dataset           # 全部train数据集
    train_iter = train_iter                 # batch之后的train数据集
    test_dataset = test_dataset             # 全部test数据集
    net = net                               # 网络模型
    loss_fn = loss_fn                       # 损失函数
    opt = opt                               # 优化器
    device = device                         # device GPU/CPU
    '''

    print('='*100)
    print(f"Runing on {device}")
    print('='*100)
    train_all_data_gpu = train_dataset.data.to(device)
    train_all_targets_gpu = train_dataset.targets.to(device)
    test_all_data_gpu = test_dataset.data.to(device)
    test_all_targets_gpu = test_dataset.targets.to(device)
    net.to(device)

    # 开始迭代
    start = time.time()
    for epoch in range(epochs):
        for batch_record in train_iter:
            X, y = batch_record                 # 分配X, y
            X, y = X.to(device), y.to(device)   # 复制到device（GPU/CPU）上
            # print(X[0])
            # print(X[0].dtype)
            # break
            y_hat = net(X)          # 计算y_hat
            loss = loss_fn(y_hat, y)# 计算loss
            opt.zero_grad()                     # 默认是累加，此处从新求导
            loss.backward()         # 计算梯度
            opt.step()              # 更新网络参数

        net.eval()  # 切换至评估模式
                    # 模型默认是net.train()
                    # 但是net中含有BN、Dropout等，在test时必须固定train时学好的参数，不能被test又改变了
                    # 但net中没有BN、Dropout等时，加不加net.eval()都无所谓

        with torch.no_grad(): # with下内容不进行grad计算，可以节省运算和内存
            train_loss = loss_fn(net(train_all_data_gpu/256), train_all_targets_gpu)
            # print(train_loss)
            train_acc_cmp = net(train_all_data_gpu/256).argmax(axis=1) == train_all_targets_gpu
            train_acc = (train_acc_cmp.sum() / len(train_acc_cmp)) * 100
            # print(train_acc)
            test_acc_cmp = net(test_all_data_gpu/256).argmax(axis=1) == test_all_targets_gpu
            test_acc = (test_acc_cmp.sum() / len(test_acc_cmp)) * 100
            # print(test_acc)
            print(f"epoch {epoch+1}/{epochs}: train_loss={train_loss}, train_acc={train_acc}, test_acc={test_acc}")
    stop = time.time()
    print('='*100)
    print(f"Total: {stop - start} sec.")
    # return (train_loss, train_acc, test_acc)
    return None

# lr 0.01 -> 0.5
# 结果表明还是会快一点收敛
net = Net()  
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.SGD(params=net.parameters(), lr=0.5)   

train_steps(epochs=10, 
            train_dataset=train_dataset, 
            train_iter=train_iter, 
            test_dataset=test_dataset, 
            net=net,                        
            loss_fn=loss_fn, 
            opt=opt, 
            device=device 
            ) 
