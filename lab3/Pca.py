import numpy as np
import sys
import cv2
from sklearn.decomposition import PCA
def max(data,maxset):
    max=sys.float_info.min
    tempt=-1
    for i in range(len(data)):
        if data[i]>max and (i not in maxset):
            tempt=i
            max=data[i]
    if tempt!=-1:
        print(tempt)
        maxset.append(tempt)
    else:
        print(max)
        print("sda")

def centralize(data):

    data_m=np.array(np.ones(data.shape[0]*data.shape[1])).reshape(data.shape[0],data.shape[1])
    data_mean=[]
    result=[]
    for i in range(data.T.shape[0]):
        data_mean.append(np.mean(data.T[i]))
        result.append(data.T[i]-np.mean(data.T[i]))
    print(data_mean)
    result=np.array(result,ndmin=2)
    # print(result.shape)
    data_mean = np.array(data_mean)
    print(data_mean)
    # show(data_mean, "mean")
    print(result.T)
    # print(type(result[0][0]))
    # #将原数据矩阵转为int8类型
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data_m[i][j]=data[i][j]

    for i in range(data_m.T.shape[0]):
        data_m.T[i]=data_m.T[i]-np.mean(data_m.T[i])
    # print(type(data_m[0][0]))
    # exit()
    #翻转一下
    return result.T,data_mean

def show(w,count):
    w=w.reshape(32,32)
    cv2.imshow(str(count),w)
    filpath="./features/"+"("+str(count)+")"+".jpg"
    cv2.imwrite(filpath, w)


def pca(X,n):
    # print(F_X)
    print(X)
    print(X.shape)
    # print(np.dot(X.T,X).shape)
    # Y=np.cov(X,rowvar=0)
    # exit()
    a,b=np.linalg.eig(np.dot(X.T,X))
    # a=1
    # b=1
    for i in range(len(a)):
        a=a.real
    d=np.array(np.ones(b.shape[0]*b.shape[1])).reshape(b.shape[0],b.shape[1])
    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            d[i][j]=b[i][j]
    print(a)
    print(b)
    # print(b[:,1])
    w=[]
    maxset=[]
    for i in range(n):
        print(i)
        max(a,maxset)
    print(maxset)
    count=0
    for i in maxset:
        w.append(d[:,i])
        # show(b[:,i],count)
        count+=1
    print(w)
    print(a)
    print(b)
    return np.array(w)










