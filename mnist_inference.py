import pickle
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
# load_mnist函数以“（训练图像，训练标签）， （测试图像， 测试标签）”
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# softmax 溢出对策
def softmax(x):
    c = np.max(x)    # 通过减去输入信号中的最大值， 来进行正确计算
    exp_x = np.exp(x - c)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x

    return y

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

print(x_train.shape)  #（60000， 784）
print(t_train.shape)  # （60000，）
print(x_test.shape)   # （10000， 784）
print(t_test.shape)   # （10000， ）

# 神经网络的输入层有784个神经元，输出层有10个神经元。输入层的784这个数字来源于图像大小的28*28=784，输出层的10这个数字来源于10类别分类
# 这个神经网络有2个隐藏层，第1个隐藏层有50个神经元，第2个隐藏层有100个神经元

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return  x_test, t_test
# normalize设置是否将输入图像正规化为0.0~1.0的值， 如果将该参数设置为false，则输入图像的像素会保持原来的0~255.
#第二个参数flatten设置是否展开输入（变成一维数组）。 如果将该参数设置为false，则输入图像为1*28*28的三维数组：设置为true，则输入图像会保存为由784个元素构成的一维数组
# 输出各个数据的形状

def init_network():
    with open(r'D:\python-project\Minist\dataset\sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)
    return network
# init_work 会读入保存在pickle文件samp——weight.pkl中的学习到的权重参数，这个文件中以字典变量的形式保存了权重和偏置参数

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = sigmoid(a3)

    return y
x, t =get_data()
network = init_network()


accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)  # 获取概率最高的元素的索引
    if p == t[i]:
        accuracy_cnt +=1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))