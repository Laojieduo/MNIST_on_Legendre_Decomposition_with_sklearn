import pandas as pd
import csv
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np

# 数据读入
idiot = np.array([[]])
flag = True
for digit in range(10):
    for csv_file in range(1, 101):
        # csv_data = pd.read_csv("./test_kmeans/"+str(digit)+"/test"+str(digit)+"_"+str(csv_file)+".csv_S_p.csv", header=None)
        # 下为从矩阵读入数据的处理方式
        file_name = "./Kmeans_6_13/"+str(digit)+"/test"+str(digit)+"_"+str(csv_file)+".csv_B.txt"
        file = open(file_name, 'r')
        # for matrixXd in range(10):
        #     count = 0
        #     sum_p_value = 0
        #     for i in range(28):
        #         for j in range(28):
        #             if csv_data.iat[28*matrixXd+i, j] > 0.00000001:
        #                 count = count + 1
        #             sum_p_value = sum_p_value + csv_data.iat[28*matrixXd+i, j]
        #     sum_p_value = sum_p_value
        #     idiot.append([sum_p_value, count])
        sum_theta = np.array([])
        sum_eta_org = 0.0
        sum_eta = np.array([])
        for line in file:
            parameters = line.strip().split(',')
            sum_theta = np.append(sum_theta, float(parameters[3]))
            sum_eta = np.append(sum_eta, float(parameters[5]))
            # sum_theta += float(parameters[3])
            # sum_eta += float(parameters[5])
        temp = np.append(sum_theta, sum_eta)
        if len(temp) < 100:
            temp = np.pad(temp, (0, 100 - len(temp)), 'constant', constant_values=0)
        temp = temp.reshape(1, 100)
        if flag==True:
            flag = False
            idiot = temp
        else:
            idiot = np.append(idiot, temp, axis=0)
print(idiot.shape)
to_pre = np.array(idiot)
print(to_pre.shape)
X = to_pre
estimator = KMeans(n_clusters=10)  # 构造聚类器
estimator.fit(X)  # 聚类
label_pred = estimator.labels_  # 获取聚类标签
print(len(label_pred))

alll = []
for i in range(10):
    this = []
    for j in range(100):
        this.append(label_pred[i*100+j])
    alll.append(this)
a = pd.DataFrame(alll)
a = pd.DataFrame(a.values.T)
result = a.apply(pd.value_counts)
result.columns = ['数字0', '数字1', '数字2', '数字3', '数字4', '数字5', '数字6', '数字7', '数字8', '数字9']
result.index = ['类别0', '类别1', '类别2', '类别3', '类别4', '类别5', '类别6', '类别7', '类别8', '类别9']
print(result)
# 绘制k-means结果

y = np.array([])
for ii in range(10):
    for count in range(100):
        y = np.append(y, int(ii))
y = y.astype(int)
label_true = y

print("AMI:", metrics.adjusted_mutual_info_score(label_true, label_pred))
print("ARI:", metrics.adjusted_rand_score(label_true, label_pred))
