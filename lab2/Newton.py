import numpy as np
import matplotlib.pyplot as plt

def fun_3(x):
    return x*x*x

def fun_2(x):
    return x*x


if __name__=='__main__':
    x = np.linspace(-10,10,1000)
    #画横线和竖线
    plt.axvline(0)
    plt.axhline(0)
    # plt.plot(x,fun_2(x),'black')
    plt.plot(x, fun_3(x), 'black')
    # plt.plot(x,2*x,'r',label="F'(x)")
    # plt.plot(x, 2 * x, 'r', label="F'(x)")
    # plt.legend()
    plt.show()