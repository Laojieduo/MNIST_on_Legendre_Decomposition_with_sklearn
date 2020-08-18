import pandas as pd
import csv
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import SpectralClustering

idiot = []
for digit in range(10):
    for csv_file in range(1, 51):
        csv_data = pd.read_csv("./test_kmeans/"+str(digit)+"/test"+str(digit)+"_"+str(csv_file)+".csv_S_p.csv", header=None)
        # file_name = "./test_kmeans/"+str(digit)+"/test"+str(digit)+"_"+str(csv_file)+".csv_B.txt"
        # file = open(file_name, 'r')
        for matrixXd in range(10):
            count = 0
            sum_p_value = 0
            for i in range(28):
                for j in range(28):
                    if csv_data.iat[28*matrixXd+i, j] > 0.00000001:
                        count = count + 1
                    sum_p_value = sum_p_value + csv_data.iat[28*matrixXd+i, j]
            sum_p_value = sum_p_value
            idiot.append([sum_p_value, count])
        # sum_theta = 0.0
        # sum_eta_org = 0.0
        # sum_eta = 0.0
        # for line in file:
        #     parameters = line.strip().split(',')
        #     # sum_theta.append(float(parameters[3]))
        #     # sum_eta_org.append(float(parameters[4]))
        #     # sum_eta.append(float(parameters[5]))
        #     sum_theta += float(parameters[3])
        #     sum_eta += float(parameters[5])
        # idiot.append([sum_theta , sum_eta])


to_pre = np.array(idiot)
X = to_pre
# 绘制数据分布图
# plt.scatter(X[:, 0], X[:, 1], c="red", marker='o', label='see')
# plt.xlabel('sum_p')
# plt.ylabel('count')
# plt.legend(loc=2)
# plt.show()
# print(type(X[0]))
estimator = SpectralClustering(n_clusters=10, gamma=1)
estimator.fit(X)  # 聚类
label_pred = estimator.labels_  # 获取聚类标签
print(len(label_pred))
alll = []
for i in range(10):
    this = []
    for j in range(50):
        this.append(label_pred[i*50+j])
    max_cluster = 0
    for digit in range(10):
        if max_cluster < this.count(digit):
            max_cluster = this.count(digit)
    print(max_cluster)
    alll = alll.append(float(max_cluster)/50.0)
print(alll)
for i in range(10):
    print(i, end=':  ')
    for j in range(50):
       print(label_pred[i*50+j], end=', ')
    print(' ', end='\n')
# a = pd.DataFrame(alll)
# a = pd.DataFrame(a.values.T)
# result = a.apply(pd.value_counts)
# result.columns = ['数字0', '数字1', '数字2', '数字3', '数字4', '数字5', '数字6', '数字7', '数字8', '数字9']
# result.index = ['类别0', '类别1', '类别2', '类别3', '类别4', '类别5', '类别6', '类别7', '类别8', '类别9' ]
# print(result)
# 绘制k-means结果
