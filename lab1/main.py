import numpy as np
import matplotlib.pyplot as plt

#对于sin(x)的生成数据函数，同时加上均值为0，方差为0.1的噪声
def data_func(x):
    y_noise = np.random.normal(0, 0.1, x.size)
    return np.array(np.sin(x)+y_noise,ndmin=2).T

#对于sin(2np.pix)的生成数据函数，同时加上均值为0，方差为0.1的噪声
def data_func_2(x):
    y_noise = np.random.normal(0, 0.1, x.size)
    return np.array(np.sin(2*np.pi*x)+y_noise,ndmin=2).T

#利用矩阵相乘得到的多项式拟合函数
def fit_func(p, x):
    X = np.array(np.ones(x.size), ndmin=2)
    for key in range(1, p.size):
        X = np.append(X, np.array(x ** key, ndmin=2), axis=0)
    X = X.T
    return np.dot(X,p)
#损失函数的实现，同样利用矩阵
def residuals_func(p, x, y):
    return 0.5*np.dot((fit_func(p, x) - y).T,(fit_func(p, x) - y))[0][0]

#利用最小二乘法求解析解，不带正则项
def analy_fit(x, y, M=0):
    X = np.array(np.ones(x.size), ndmin=2)
    for key in range(1, M+1):
        X = np.append(X, np.array(x ** key, ndmin=2), axis=0)
    X = X.T
    X_1= X.T.dot(X)
    X_2 = np.linalg.inv(X_1)#求逆
    X_3=X_2.dot(X.T)
    X_4=X_3.dot(y)
    p = np.array(X_4,ndmin=2)
    return p
#加上正则项的解析解
def analy_fit_de(x, y,t, M=0):
    I = np.identity(M+1)
    print(I)
    X = np.array(np.ones(x.size), ndmin=2)
    y = np.array(y,ndmin=2)
    for key in range(1, M+1):
        X = np.append(X, np.array(x ** key, ndmin=2), axis=0)
    X = X.T
    X_1= X.T.dot(X)+t*I
    X_2 = np.linalg.inv(X_1)
    X_3=X_2.dot(X.T)
    X_4=X_3.dot(y)
    p = np.array(X_4,ndmin=2)
    print(p)
    return p

#求解梯度，同样利用矩阵乘实现
def grad(x,y,p):
    return np.dot(x.T,np.dot(x,p)-y)/y.size

#线搜索寻找步长
def line_search(x,y,p,g):
    alpha = 4
    c = 0.8
    ro = 0.5
    X = np.array(np.ones(x.size), ndmin=2)
    for key in range(1, p.size):
        X = np.append(X, np.array(x ** key, ndmin=2), axis=0)
    X = X.T
    while residuals_func(p+alpha*g,x,y)>residuals_func(p,x,y)+c*alpha*np.dot(grad(X,y,p).T,g):
        alpha = ro*alpha
    return alpha
#共轭梯度法的实现
def conjuate_grad(x,y,M=0):
    p = np.array(np.random.rand(M + 1), ndmin=2).T
    X = np.array(np.ones(x.size), ndmin=2)
    for key in range(1, M + 1):
        X = np.append(X, np.array(x ** key, ndmin=2), axis=0)
    X = X.T
    g0 = grad(X,y,p)
    d0 = -g0
    count=1
    while count<=3000:
        p = p + line_search(x,y,p,d0)*d0
        g1 = grad(X,y,p)
        beta = np.dot(g1.T,g1)/np.dot(g0.T,g0)
        d0 = -g1 + beta*d0
        g0 = g1
        count+=1
    return p

#梯度下降法
def gradient_func(x,y,a,n,M=0):
    count=1
    X=np.array(np.ones(x.size),ndmin=2)
    for key in range(1,M+1):
        X = np.append(X,np.array(x**key,ndmin=2),axis=0)
    X=X.T
    p = np.array(np.random.rand(M+1),ndmin=2).T
    # while abs(residuals_func(p,x,y)>1e-5):
    while abs(residuals_func(p,x,y))>1e-5:
        print(count)
        p = p-a*grad(X,y,p)
        count=count+1
    print("count=",count)
    return p


#主函数
if __name__ == '__main__':
    choice = int(input())
    m = []
    res = []
    res_predit=[]
    M = 1
    t = 0.1
    x = np.linspace(0, 1, 100)
    x_pre = np.linspace(1,2,100)
    y = data_func_2(x)
    y_pre = np.sin(2*np.pi*x_pre)
    x_points = np.linspace(0, 2, 1000)
    plt.figure(1)
    plt.plot(x,y,'ob')
    plt.plot(x_points,np.sin(2*np.pi*x_points))
    fig, axes = plt.subplots(3, 4, figsize=(5, 5))

    print(x)
    print(y)
    N = 100000
    count=0
    flag=0
    while (M <= 12):
        if choice==1:
            print('1')
            p=analy_fit(x,y,M)
        elif choice==2:
            print('2')
            p=analy_fit_de(x,y,0.3,M)
        elif choice==3:
            print('3')
            p=gradient_func(x,y,t,N,M)
        elif choice==4:
            print('4')
            p=conjuate_grad(x,y,M)
        else:
            print("wrong input")
            exit()
        #画图
        axes[flag][count].plot(x_points, fit_func(p, x_points), label="fit")
        axes[flag][count].plot(x_points, np.sin(2*np.pi*x_points), label='sinx')
        string = "M=" + str(M)
        axes[flag][count].set_title(string)
        m.append(M)
        res.append(abs(residuals_func(p, x, y)))
        res_predit.append(abs(residuals_func(p,x_pre,y_pre)))
        M = M + 1
        count = count + 1
        if count == 4:
            flag = flag + 1
            count = 0
        t += 0.1
        N+=10000
    plt.figure(3)
    plt.plot(m, res)
    plt.figure(4)
    plt.plot(m,res_predit)
    plt.title("res-m")
    plt.show()
