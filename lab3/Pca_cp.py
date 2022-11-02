import numpy as np
import cv2 as cv
import math
import matplotlib.pyplot as plt
from sklearn import decomposition

# 数据中心化
def centere_data(dataMat):
    rows, cols = dataMat.shape
    meanVal = np.mean(dataMat, axis=0)  # 按列求均值，即求各个特征的均值
    meanVal = np.tile(meanVal, (rows, 1))#扩大矩阵
    newdata = dataMat - meanVal
    print(newdata)
    return newdata, meanVal

# 最小化降维造成的损失，确定k
def Percentage2n(eigVals, percentage):
    sortArray = np.sort(eigVals)  # 升序
    sortArray = sortArray[-1::-1]  # 逆转，即降序
    arraySum = sum(sortArray)
    temp_Sum = 0
    num = 0
    for i in sortArray:
        temp_Sum += i
        num += 1
        if temp_Sum >= arraySum * percentage:
            return num


# 得到最大的k个特征值和特征向量
def EigDV(covMat, k):
    D, V = np.linalg.eig(covMat)  # 得到特征值和特征向量
    # k = Percentage2n(D, p)  # 确定k值
    print("降维后的特征个数：" + str(k) + "\n")
    eigenvalue = np.argsort(D)
    K_eigenValue = eigenvalue[-1:-(k + 1):-1]
    K_eigenVector = V[:, K_eigenValue]
    return K_eigenValue, K_eigenVector

def SNR(img_raw,img):
    sum1=0
    sum2=0
    print(img)
    print(img_raw)
    # exit()
    img=np.array(img)
    img_raw=np.array(img_raw)
    for i in range(img_raw.shape[0]):
        for j in range(img_raw.shape[1]):
            sum1+=img_raw[i][j]**2
            print(sum1)
            sum2+=(img_raw[i][j]-img[i][j])**2
    return math.log(sum1/sum2,10)*10


# PCA算法
def PCA(data, k):
    dataMat = np.float32(np.mat(data))
    # 数据中心化
    dataMat, meanVal = centere_data(dataMat)
    # 计算协方差矩阵
    print(dataMat.shape)
    covMat = np.cov(dataMat, rowvar=0)
    print(covMat.shape)
    # 选取最大的k个特征值和特征向量
    D, V = EigDV(covMat, k)
    # 得到降维后的数据
    lowDataMat = dataMat * V
    print(lowDataMat.shape)
    # 重构数据
    reconDataMat = lowDataMat *V.T+ meanVal
    return reconDataMat

if __name__ == '__main__':
    img = cv.imread('./gray_data/1.jpg',0)
    rows, cols = img.shape
    print(img)
    #pca = decomposition.PCA()
    print("降维前的特征个数：" + str(cols) + "\n")
    print(img)
    print('----------------------------------------')
    k=[]
    loss=[]
    for i in range(40):
        k.append(i+1)
        PCA_img = PCA(img, i+1)
        print(PCA_img.shape)
        PCA_img = PCA_img.astype(np.uint8)
        print(PCA_img)
        loss.append(SNR(img,PCA_img))
        cv.imshow('test', PCA_img)
        cv.imwrite("./testimage/"+"("+str(i+1)+")"+".jpg",PCA_img)
        # cv.waitKey(0)
    cv.destroyAllWindows()
    plt.plot(k,loss)
    plt.show()




