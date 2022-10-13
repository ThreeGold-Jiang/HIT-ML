import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

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