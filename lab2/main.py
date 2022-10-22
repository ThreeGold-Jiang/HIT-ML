import numpy as np
import matplotlib.pyplot as plt

#手工生成数据集
def dataset_make(m, n, k, b):
    mean = np.random.randn(n)
    cov = np.eye(n)
    size = (m)
    #协方差矩阵表示相关性
    cov=[[1,0],
         [0,1]]
    x = np.random.multivariate_normal(mean, cov, size)
    label = np.random.randint(0, 2, (m, 1))
    for i in range(m):
        if x[i][1] > k * x[i][0] + b:
            label[i][0] = 1
        else:
            label[i][0] = 0
    print(label)
    print(x)
    return x, label

#绘制散点图
def draw(x, y,p=plt):
    x_1 = []
    x_2 = []
    x_3 = []
    x_4 = []
    for i in range(x.shape[0]):
        if y[i][0] == 1:
            x_1.append(x[i][0])
            x_2.append(x[i][1])
        else:
            x_3.append(x[i][0])
            x_4.append(x[i][1])
    p.scatter(x_1, x_2, c='red')
    p.scatter(x_3, x_4, c='blue')

#逻辑回归类
class logisit:
    def __init__(self, length):
        self.happy = 1
    #初始化标签
        self.w = np.array(np.ones(length)*0.001, ndmin=2).T
    #逻Sigmod函数
    def module(self, x):
        p = np.dot(x, self.w)
        p_1 = 1 / (1 + np.exp(-p))
        return p_1
    #极大似然估计函数
    def likelihood(self, x, y):
        re = (y * np.log(self.module(x)) + (1 - y) * np.log(1 - self.module(x))) / x.shape[0]
        b = np.array(np.ones(y.shape[0]), ndmin=2)
        # 正则项
        # 取负值就是求极大似然函数的最小值
        return np.dot(b, re)[0][0]
    #梯度函数
    def grad(self, x, y):
        return np.dot(x.T, self.module(x) - y) / x.shape[0]
    #加入正则项的梯度函数
    def grad_1(self, x, y):
        return np.dot(x.T, self.module(x) - y) / x.shape[0]+self.w
    #求二阶黑塞矩阵
    def grad_2(self, x, y):
        g = np.ones(x.shape[1] * x.shape[1]).reshape((x.shape[1], x.shape[1]))
        # print(x.shape[1])
        for i in range(x.shape[1]):
            for j in range(x.shape[1]):
                b = np.array(np.ones(x.shape[0]), ndmin=2)
                p = np.dot(x, self.w)
                if i==j:
                    g[i][j] = np.dot(b, (x.T[i]).T * (x.T[j]).T * (self.module(x) ** 2) * np.exp(-p))[0][0] / x.shape[0] + 1
                else:
                    g[i][j] = np.dot(b, (x.T[i]).T * (x.T[j]).T * (self.module(x) ** 2) * np.exp(-p))[0][0] / x.shape[0]
        return g
    #采用欧式距离的Loss函数
    def loss(self, x, y,w):
        d = np.ones(y.shape[0])
        return np.dot(d,np.abs(self.module(x)-y))
    #更新w
    def set_w(self, w):
        self.w = w

#梯度下降函数
def grad_func(features, labels, func, a, n):
    count = 0
    #插入一列1，用于迭代b
    b = np.ones(features.shape[0])
    x = np.insert(features, features.shape[1], values=b, axis=1)
    y = labels
    print('grad1.0')
    while count <= n:
        print(count, '/', str(n), func.loss(x, y,func.w), func.likelihood(x, y))
        count += 1
        func.set_w(func.w - a * func.grad(x, y))
    return func.w

#带正则的梯度下降
def grad_func_1(features, labels, func, a, n):
    count = 0
    b = np.ones(features.shape[0])
    x = np.insert(features, features.shape[1], values=b, axis=1)
    y = labels
    while count <= n:
        print(count, '/', str(n), func.loss(x, y,func.w), func.likelihood(x, y))
        count += 1
        func.set_w(func.w - a* func.grad_1(x, y))
    return func.w

# 线搜索寻找步长
def line_search(x,y,func):
    alpha = 1
    c = 0.8
    ro = 0.5
    while func.loss(x,y,func.w -alpha*np.dot(np.linalg.inv(func.grad_2(x, y)), func.grad_1(x, y)))>func.loss(x,y,func.w)+c*alpha*np.dot(func.grad_1(x,y).T,func.w)+1e-3:
        alpha = ro*alpha
    return alpha

#牛顿法
def newton_1(features, labels, func, a,n):
    count = 0
    b = np.ones(features.shape[0])
    x = np.insert(features, features.shape[1], values=b, axis=1)
    y = labels
    print('grad1.0')
    while count <= n:
        print(count, '/',str(n), func.loss(x, y,func.w), func.likelihood(x, y))
        count += 1
        func.set_w(func.w -a*np.dot(np.linalg.inv(func.grad_2(x, y)), func.grad_1(x, y)))
    return func.w


if __name__ == '__main__':
    #数据集
    x, y = dataset_make(500, 2, 10, 1)
    #绘图
    a=plt.figure(1)
    draw(x, y)
    fig,axes = plt.subplots(2, 3, figsize=(10, 10))
    j=0
    for i in range(0,6):
        func=logisit(x.shape[1] + 1)
        w=grad_func(x, y,func,0.01,10000*i)
        x_2 = np.linspace(-5, 5, 1000)
        print(w.shape)
        print(x.shape)
        y_2 = (-x_2 * func.w[0][0] + func.w[2][0]) / func.w[1][0]
        if i < 3:
            axes[j][i].plot(x_2, y_2)
            draw(x, y,axes[j][i])
            axes[j][i].set_title(str(10000*i))
        else:
            j=1
            axes[j][i-3].plot(x_2, y_2, 'black')
            draw(x, y,axes[j][i-3])
            axes[j][i-3].set_title(str(10000*i))
    plt.show()
