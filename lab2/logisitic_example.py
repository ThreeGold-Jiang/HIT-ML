import matplotlib.pyplot as plt
import numpy as np

def unite_step(x):
    y=np.array(np.zeros(x.shape[0]))
    for i in range(x.shape[0]):
        if x[i] > 0:
            y[i]=1
        elif x[i] < 0:
            y[i]=0
        else:
            y[i]=0.5
    return y

x = np.linspace(-10,10,10000)
print(x.shape)
y = np.exp(x)/(1+np.exp(x))
y_2 = unite_step(x)
plt.plot(x,y,'b',label='sigmod')
plt.plot([0],[0.5],'ro')
x = np.linspace(-10,-0.1,10000)
plt.plot(x,unite_step(x),'r')
x = np.linspace(0.1,10,10000)
plt.plot(x,unite_step(x),'r',label='united_step')
plt.legend()
plt.show()
