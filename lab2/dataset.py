import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

def dataset_from_skylearn():
    #乳腺癌威斯康星（诊断）数据集[二分类预测]
    cancer_data_bunch = load_breast_cancer()
    print('features',cancer_data_bunch.feature_names)
    print('labels',cancer_data_bunch.target_names)#良性肿瘤/恶性肿瘤

    cancer_data = pd.DataFrame(cancer_data_bunch.data)
    cancer_target = pd.DataFrame(cancer_data_bunch.target)
    print(cancer_data)
    # print(cancer_data.shape[0])
    cancer_data.insert(loc=cancer_data.shape[1],column=30,value=cancer_target)
    cancer_data.to_csv('./data/dataset.txt',sep='\t',index=False,header=False)
    # fea=np.ndarray(cancer_data)
    # print(fea)


def dataset_make(m,n,k,b):
    mean = np.random.randn(n)
    cov = np.eye(n)
    size = (m)
    for i in range(n):
        cov[i][i]=1
    x=np.random.multivariate_normal(mean,cov,size)
    label=np.random.randint(0,2,(m,1))
    for i in range(m):
        if x[i][1]>k*x[i][0]+b:
            label[i][0]=1
        else:
            label[i][0]=0
    print(label)
    print(x)

    return x,label


class data_get:
    def __init__(self, source):
        self.source = source
        self.features = []
        self.labels = []
        self.num = 0
        self.build(source)

    def normalize(self):
        data = self.features.T
        for i in range(data.shape[0] - 1):
            max = np.max(data[i])
            min = np.min(data[i])
            # print(max)
            # print(min)
            for j in range(data.shape[1]):
                data[i][j] = (data[i][j] - min) / (max - min)
        self.features = data.T

    def build(self, source):
        with open(source, 'r') as fp:
            data_line = fp.readline()
            while data_line != '':
                data_set = data_line.split('\t')
                data = []
                label = 0
                for k in range(len(data_set)):
                    if data_set[k] != '\n' and k <= len(data_set) - 3:
                        data.append(float(data_set[k]))
                    if data_set[k] == '1\n':
                        label = int(1)
                    if data_set[k] == '0\n':
                        label = int(0)
                self.features.append(data)
                self.labels.append(label)
                data_line = fp.readline()
            self.features = np.array(self.features, ndmin=2)
            b = np.ones(self.features.shape[0])
            self.num = self.features.shape[1] + 1
            self.features = np.insert(self.features, self.features.shape[1], values=b, axis=1)
            self.labels = np.array(self.labels, ndmin=2).T


    def draw(self, i, j,p=plt):
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
        p.scatter(x_1, y_1, c='red')
        p.scatter(x_2, y_2, c='blue')
