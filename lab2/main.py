import numpy as np
import matplotlib.pyplot as plt
class data_get:
    def __init__(self,source):
        self.source=source
        self.features=[]
        self.labels=[]
        self.num=0
        self.build(source)

    def normalize(self):
        data=self.features.T
        for i in range(data.shape[0]-1):
            max=np.max(data[i])
            min=np.min(data[i])
            # print(max)
            # print(min)
            for j in range(data.shape[1]):
                data[i][j]=(data[i][j]-min)/(max-min)
        self.features=data.T

    def build(self,source):
        with open(source,'r') as fp:
            data_line=fp.readline()
            while data_line!='':
                data_set = data_line.split('\t')
                # print(data_set)
                data = []
                label = 0
                for k in range(len(data_set)):
                    if data_set[k] != '\n' and k <= len(data_set) - 3:
                        data.append(float(data_set[k]))
                    if data_set[k] == '1\n':
                        label = int(1)
                    if data_set[k] == '0\n':
                        label = int(0)
                # print(data)
                # print(label)
                self.features.append(data)
                self.labels.append(label)
                data_line = fp.readline()
            # print(self.features)
            # print(self.labels)
            self.features= np.array(self.features,ndmin=2)
            #添加b（属性）行，取值全为1
            b = np.ones(self.features.shape[0])
            self.num=self.features.shape[1]+1
            self.features=np.insert(self.features,self.features.shape[1],values=b,axis=1)
            # print(self.features.shape)
            # print(f.shape)
            self.labels=np.array(self.labels,ndmin=2).T
            # print(l.shape)
            # print(l[0][0])
            #lamda
    def draw(self,i,j):
        x = self.features.T[i]
        y = self.features.T[j]
        x_1 = []
        y_1 = []
        x_2 = []
        y_2 = []
        for k in range(x.shape[0]):
            if self.labels[k][0] == 0:
                x_1.append(x[k])
                y_1.append(y[k])
            else:
                x_2.append(x[k])
                y_2.append(y[k])
        plt.scatter(x_1, y_1, c='red')
        plt.scatter(x_2, y_2, c='blue')
        # print(x.shape)
        # plt.plot(x,y,'rd')
        plt.show()

class logisit:
    def __init__(self,length):
        self.happy=1
        self.w=np.array(np.ones(length)*0.0001,ndmin=2).T
    def module(self,x):
        p_1 = 1/(1+np.exp(-np.dot(x,self.w)))
        # p_1=np.dot(x,self.w)
        return p_1
    def likelihood(self,x,y):
        #要最大似然估计最大，即它的负值最小
        # re = np.dot(x,self.w)*y-np.log(1+np.exp(np.dot(x,self.w)))
        re = y*self.module(x)+(1-y)*np.log(1-self.module(x))
        L=0
        for i in range(re.shape[0]):
           L+=re[i][0]
        # 正则项
        # L = L + (np.dot(self.w.T,self.w)[0][0])**2*(1/2)
        #取负值就是求极大似然函数的最小值
        return L
    def grad(self,x,y):
        return np.dot(x.T,self.module(x)-y)
    def predict(self,x):
        p=np.floor(self.module(x)+0.5)
        return p
    def loss(self,x,y):
        d = np.ones(y.shape[0])
        # print(self.predict(x))
        return np.dot(d,np.abs(self.module(x)-y))


def grad_func(data,func,a):
    count=0
    x=data.features[0:30]
    y=data.labels[0:30]
    print('grad1.0')
    # while func.loss(data.features,data.labels)>=0:
    # while func.likelihood(data.features,data.labels).all()>-3000:
    while count<=300:
        count+=1
        func.w=func.w-a*func.grad(x,y)
        print(func.grad(x,y).T)
        print(count,func.loss(x,y),func.likelihood(x,y))


if __name__=='__main__':
    data=data_get('./data/dataset.txt')
    # print(data.labels)
    # print(data.features)
    # print(data.num)
    log=logisit(data.num)
    # print(data.features)
    # data.normalize()
    # print(data.features)
    # print(log.module(data.features))
    # print(data.features)
    # print(log.w)
    # print(type(log.w[0][0]))
    # print(np.exp(4000))
    #溢出来
    #print(np.exp(400))
    print('grad')
    print(log.module(data.features))
    print(log.grad(data.features,data.labels))
    # grad_func(data,log,0.111111)
    print(log.likelihood(data.features,data.labels))
    print(data.labels*log.module(data.features))
    data_1 = data_get('./data/dataset.txt')
    #
    # print(log.loss(data_1.features,data_1.labels))
    # print(log.w)
    # x=np.linspace(0, 1000, num=len(data.labels))
    # data.draw(2,28)



