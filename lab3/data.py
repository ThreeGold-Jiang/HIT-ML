import numpy as np
import cv2
import math
import scipy
from scipy.io import loadmat
def data_make(mean,cov,size):
    x=np.random.multivariate_normal(mean,cov,size)
    return x


def img_get2(X,n,m):
    # data = scipy.io.loadmat(r".\Yale_32x32.mat")
    # # 加载后的数据是字典类型
    # print(type(data))
    # # 使用。keys()查看字典的键
    # print(data.keys())
    # X = data['fea']
    print(X.shape)
    for i in range(0,1):
        filepath2="./result/"+str(m)+"("+str(i+1)+")"+".jpg"
        print(filepath2)
        print(X.shape)
        img=X[i].reshape(int(math.sqrt(n)),int(math.sqrt(n)))
        cv2.imshow("1",img)
        lp1 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        # cv2.imshow("2",img)
        # img = cv2.imread(filepath2, cv2.IMREAD_GRAYSCALE)
        print(img.shape)
        cv2.imwrite(filepath2,img)

def img_get(filepath,n):
    # img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    # filepath_2="./gray_data/"
    # count=1
    # for i in range(1,301):
    #     filepath1=filepath+"("+str(i)+")"+".jpg"
    #     filepath2=filepath_2+"("+str(i)+")"+".jpg"
    #     print(filepath1)
    #     img = cv2.imread(filepath1, cv2.IMREAD_GRAYSCALE)
    #     cv2.imwrite(filepath2,img)

    imgs=[]
    for i in range(1,166):
        filepath1=filepath+"("+str(i)+")"+".jpg"
        img = cv2.imread(filepath1,cv2.IMREAD_GRAYSCALE)
        print(img.shape)
        # cv2.imshow(str(i),img)
        # cv2.waitKey(0)
        #
        # print(img)
        # exit(1)
        img=img.reshape(img.shape[0]*img.shape[1])
        imgs.append(img)
        print(img)
    # img1 = cv2.imread(filepath,cv2.IMREAD_COLOR)
    imgs=np.array(imgs,ndmin=2)
    # cv2.imshow('image',img)
    cv2.waitKey(n)

    return imgs









