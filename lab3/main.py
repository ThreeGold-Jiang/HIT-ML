import matplotlib.pyplot as plt
import numpy as np

import Pca as pc
import data as data
import draw as dr
import cv2
def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    print_hi('HI,JJY')
    mean=[0,0,0]
    cov=[[100,1,11],[1,100,1],[11,1,0.000001]]
    size=(1000)
    '''
    实验验证第一部分
    '''
    # dataset=data.data_make(mean,cov,size)
    # dr.draw_3d_scatter(dataset.T[0],dataset.T[1],dataset.T[2],'b','o',"test",1)
    # dr.draw_2d_scatter(dataset.T[1], dataset.T[2], 'red', 'o', "test2",2)
    # print(dataset)
    # dataset = np.array(dataset)
    # print(dataset.shape)
    # dataset,data_mean=pc.centralize(dataset)
    # print(dataset.shape)
    # pca_dataset,b=pc.pca(dataset,2)
    # # print(np.dot(dataset,pca_dataset.T))
    # result=np.dot(dataset,pca_dataset.T)
    # print(result.shape)
    # result_1=np.dot(dataset,b.T)
    # dr.draw_3d_scatter(result_1.T[0], result_1.T[1], result_1.T[2], 'b', 'o', "test",4)
    # dr.draw_2d_scatter(result.T[0],result.T[1],'b','o',"test",3)
    '''
    实验验证第二部分
    '''
    filepath = r"./dataset/"
    dataset=data.img_get(filepath,1)
    # for i in range(dataset.shape[0]):
    #     pc.show(dataset[i],i)
    # exit()
    print(dataset.shape)
    data_cen,data_mean=pc.centralize(dataset)
    # print(pc.centralize(dataset))
    # print(dataset)
    data_mean = np.tile(data_mean, (dataset.shape[0], 1))
    for i in range(100):
        result = pc.pca(data_cen, i+1)
        print(result.shape)
        print(data_cen.shape)
        print(data_cen)
        print(result)
        # exit()
        print(data_mean.shape)
        # 逆中心化
        # data_cen=data_cen+data_mean.T
        # X=np.uint8(np.dot(data_cen,result)+255)
        X = np.dot(np.dot(data_cen, result.T), result) + data_mean
        data.img_get2(X, 1024,i+1)
    # print(data_mean.shape)
    # print(X.shape)
    # print(result.shape)
    # exit()



    # data.img_get2("2s")

    # print(np.dot(dataset.T,pc.pca(dataset,60).T).shape)

    # cv2.imshow("im1",dataset)
    # cv2.imshow("im2",ls1)
    # cv2.waitKey(0)
    # pc.pca(dataset,1120)
    # print(pc.pca(dataset,120).shape)


    # print(np.dot(dataset.T,dataset)/100)
    # dataset=np.dot(pc.pca(dataset,2),dataset.T).T
    # print(dataset)
    plt.show()




# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
