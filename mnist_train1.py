import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pylab as plt
# matplotlib_绘制二维和三维图表的数据可视化工具; pyplot_面向当前图
from dataset.mnist import load_mnist

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# sigmoid函数作为激活函数，激活函数的作用在于决定如何来激活输入信号的总和

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)  # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))
# softmax 溢出对策
#如上所示，softmax函数的输出是0到1.0之间的实数，softmax函数的输出值的总和是1
#输出总和为1是softmax函数的一个重要特质
#把softmax函数的输出解释为“概率”
#通过使用softmax函数，可以用概率的（统计）方法处理问题

#min-batch版交叉熵误差的实现
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
#np.arange([start, ]stop, [step, ]dtype=None)
#start:起点值；可忽略不写，默认从0开始
#stop:终点值；生成的元素不包括结束值
#step:步长；可忽略不写，默认步长为1
#dtype:默认为None，设置显示元素的数据类型

def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 还原值
        it.iternext()

    return grad


class ThreeLayerNet:
    def __init__(self, input_size, hidden_size, mid_size, output_size, weight_init_std=0.01):
        print("Build Net")
        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, mid_size)
        self.params['b2'] = np.zeros(mid_size)
        self.params['W3'] = weight_init_std * np.random.randn(mid_size, output_size)
        self.params['b3'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2, W3 = self.params['W1'], self.params['W2'], self.params['W3']
        b1, b2, b3 = self.params['b1'], self.params['b2'], self.params['b3']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = sigmoid(a2)
        a3 = np.dot(z2, W3) + b3
        y = softmax(a3)

        return y

    # 损失函数
    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)


    # 数值微分法
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        grads['W3'] = numerical_gradient(loss_W, self.params['W3'])
        grads['b3'] = numerical_gradient(loss_W, self.params['b3'])
        return grads


    # 误差反向传播法
    def gradient(self, x, t):
        W1, W2, W3 = self.params['W1'], self.params['W2'], self.params['W3']
        b1, b2, b3 = self.params['b1'], self.params['b2'], self.params['b3']
        grads = {}


        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = sigmoid(a2)
        a3 = np.dot(z2, W3) + b3
        y = softmax(a3)


        # backward
        dy = (y - t) / batch_num
        grads['W3'] = np.dot(z2.T, dy)
        grads['b3'] = np.sum(dy, axis=0)

        da2 = np.dot(dy, W3.T)
        dz2 = sigmoid_grad(a2) * da2
        grads['W2'] = np.dot(z1.T, dz2)
        grads['b2'] = np.sum(dz2, axis=0)

        da1 = np.dot(dz2, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads

    # 准确率
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

if __name__ == '__main__':
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    network = ThreeLayerNet(input_size=784, hidden_size=50, mid_size=100,  output_size=10)



    train_loss_list = []

    # 训练和验证
    batch_size = 200
    iter_nums = 20000
    train_size = x_train.shape[0]
    learning_rate = 0.1

    # 记录准确率
    train_acc_list = []
    test_acc_list = []
    # 平均每个epoch的重复次数
    iter_per_epoch = max(train_size / batch_size, 1)

    for i in range(iter_nums):
        # 小批量数据
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # 计算梯度
        # 数值微分 计算很慢
        # grad=net.numerical_gradient(x_batch,t_batch)
        # 误差反向传播法 计算很快
        grad = network.gradient(x_batch, t_batch)

        # 更新参数 权重W和偏重b
        for key in ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']:
            network.params[key] -= learning_rate * grad[key]

        # 记录学习过程
        loss = network.loss(x_batch, t_batch)
        print('训练次数:' + str(i) + '    loss:' + str(loss))
        train_loss_list.append(loss)

        # 计算每个epoch的识别精度
        if i % iter_per_epoch == 0:
            # 测试在所有训练数据和测试数据上的准确率
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print('train acc:' + str(train_acc) + '   test acc:' + str(test_acc))

    print(train_acc_list)
    print(test_acc_list)

    # 绘制图形
    markers = {'train': 'o', 'test': 's'}
    x = np.arange(len(train_acc_list))
    plt.plot(x, train_acc_list, label='train acc')
    plt.plot(x, test_acc_list, label='test acc', linestyle='--')
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.show()
