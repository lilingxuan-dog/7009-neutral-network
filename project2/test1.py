import torch
import numpy as np
# 加载训练和测试数据，将训练数据分为训练集和验证集
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import time

#加载数据
num_workers = 0
#每批加载16张图片
batch_size = 16
# percentage of training set to use as validation
valid_size = 0.2

#将数据转换为torch.FloatTensor,并标准化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

#选择对应训练集与测试集的数据
train_data = datasets.CIFAR10(
    'data',train=True,
    download=True,transform=transform
)
test_data = datasets.CIFAR10(
    'data',train=False,download=True,transform=transform
)

#obtain training indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int (np.floor(valid_size*num_train))
train_idx,valid_idx = indices[split:],indices[:split]

#define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

#对数据进行装载，利用batch_size来确认每个包的大小
train_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size,
                                           sampler=train_sampler,num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size,
                                           sampler=valid_sampler,num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data,batch_size=batch_size,
                                          num_workers=num_workers)

#图像分类为10个类别
img_class = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']


examples = iter(test_loader)
images, labels = examples.__next__()

import torch.nn as nn
import torch.nn.functional as F

# 搭建卷积神经网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        # 卷积层&池化层
        #32x32x3
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, stride=1,padding=1)

        #16x16x64
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride=1,padding=1)

        #8x8x128
        self.conv3 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride=1,padding=1)
        #池化层
        self.pool = nn.MaxPool2d(2,2)
        #全连接层
        self.fc1 = nn.Linear(4 * 4 * 256, 512) # 输入4*4*256， 输出512
        self.fc2 = nn.Linear(512, 256) # 输入512， 输出256
        self.fc3 = nn.Linear(256, 10) # 输入256， 输出10（分10类）
        self.dropout = nn.Dropout(0.5)


    #前向计算
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # print("x_shape:", x.size())
        x = x.view(x.size()[0], -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# 实例化模型
model = CNN()
print(model)

# 设置模型的训练次数
n_epochs = 30
# 定义损失函数->交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# 定义优化方法->
optimizer = optim.SGD(model.parameters(), lr=0.01)

train_losses = []
train_acces = []
eval_losses = []
eval_acces = []

print("start training! \nplease wait for minutes·····")
for epoch in range(n_epochs):
    train_loss = 0.0
    train_acc = 0.0

    # 记录训练开始时刻
    start_time = time.time()

    ##### 训练集的模型 #####
    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss


    # 计算平均值
    train_losses.append(train_loss / len(train_loader))

    # 验证集的模型 #
    eval_loss = 0.0
    eval_acc = 0.0

    model.eval()
    for data, target in valid_loader:
        output = model(data)
        loss = criterion(output, target)
        eval_loss += loss

    # 计算平均值
    eval_acces.append(train_acc / len(test_loader))

    # 输出显示训练集与验证集的损失函数
    print('epoch:{}, Train Loss:{:.5f}, '
          'Test Loss:{:.5f}'
          .format(epoch, train_loss / len(train_loader),
                  eval_loss / len(test_loader),
                  ))
    stop_time = time.time()
    print("time is:{:.4f}s".format(stop_time - start_time))


# 测试模型
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

model.eval()
for data, target in test_loader:
    output = model(data)
    loss = criterion(output, target)
    test_loss += loss
    _, pred = torch.max(output, 1)
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy())
    for i in range(batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

test_loss = test_loss/len(test_loader)
print('Test Loss:{:.5f}\n'.format(test_loss))
print("End train.")

for i in range(10):
    if class_total[i] > 0:
        print('Test Acc of %5s: %2d%% (%2d/%2d)' % (
            img_class[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Acc of %5s: N/A (no training examples)' % (img_class[i]))

print('\nTest Acc (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))
