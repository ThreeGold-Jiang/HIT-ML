import numpy as np
import matplotlib.image
from PIL import Image
import imageio
from skimage import io, transform, color


#图片转矩阵
def image_matrix(filename):
    x = Image.open(filename)
    data = np.asarray(x)
    #print(data.shape)#输出图片尺寸
    #print(data)#输出图片矩阵
    return data
#主成分分析法
#输入：矩阵X
#返回：投影矩阵（按唯独的重要性顺序）、方差、均值
def pca(X):
    #获取维数
    num_data, dim = X.shape
    #数据中心化
    mean_X = X.mean(axis=0)
    X = X - mean_X

    if dim > num_data:
        #PCA 使用紧致技巧
        M = np.dot(X,X.T)#协方差矩阵
        e,EV = np.linalg.eigh(M)#特征值和特征向量
        tmp = np.dot(X.T,EV).T#紧致技巧
        V = tmp[::-1]#由于最后的特征向量是按我们所需要的，所以需要将其逆转
        S = np.sqrt(e)[::-1]#由于特征值是按照递增顺序排列的，所以需要将其逆转
        for i in range(V.shape[1]):
            V[:,i] /= S
    else:
        #PCA 使用SVD方法
        U,S,V = np.linalg.svd(X)
        V = V[:num_data]#仅仅返回前num_data维的数据才合理
        #返回投影矩阵、方差和均值
    return V#,S,mean_X




#矩阵转图片
def matrix_image(matrix):
    im = Image.fromarray(matrix)
    return im



#整个PCA算法：输出主成分提取之后的图片
def PCA_hole(filename):
    img = image_matrix(filename)
    low_matrix = pca(img)#将图片进行PCA运算
    print(low_matrix.shape)
    print(low_matrix)
    end_matrix = low_matrix *255#运算结果乘255
    end_img = matrix_image(end_matrix)#乘255之后的矩阵转为图片
    return end_matrix#返回处理后的图片
'''#单个图片PCA
img = image_matrix('D:\\个人资料\\数据集\\微博评论采集表情+图片\\灰度图unit8\\train\\新建文件夹I1.png')#获取图片并转维矩阵
#print(img)
#print("***************************************")
low_matrix = pca(img)#将图片进行PCA运算
#print(low_matrix)
#print(".............")
end_matrix = low_matrix * 255#运算结果乘255
end_img = matrix_image(end_matrix)#乘255之后的结果转维图片
end_img.show()#显示图片
'''
#主程序：保存PCA之后的图片
datapath = './dataset'#图片所在路径
str = datapath + '/*.jpg'#识别.png图像
coll = io.ImageCollection(str,load_func = PCA_hole)#批处理
for i in range(len(coll)):
    print(coll[i].shape)
    io.imsave('./result/'+ np.str(i) + '.jpg',coll[i])#保存处理后的图片
    #print(coll[i])#输出每一个pca之后的矩阵
