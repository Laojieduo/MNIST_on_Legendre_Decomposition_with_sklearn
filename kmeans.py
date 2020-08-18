import pandas as pd
from sklearn import manifold
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import savefig

idiot = np.array([[]])
flag = True
for digit in range(10):
    for csv_file in range(1, 101):
        # csv_data_p = pd.read_csv("./Kmeans_6_18/"+str(digit)+"/test"+str(digit)+"_"+str(csv_file)+".csv_S_p.csv", header=None)
        # csv_data_theta = pd.read_csv("./Kmeans_6_18/"+str(digit)+"/test"+str(digit)+"_"+str(csv_file)+".csv_theta.csv", header=None)
        # csv_data_eta = pd.read_csv("./Kmeans_6_18/"+str(digit)+"/test"+str(digit)+"_"+str(csv_file)+".csv_eta.csv", header=None)
        file_name = "./Kmeans_6_18/"+str(digit)+"/test"+str(digit)+"_"+str(csv_file)+".csv_DKL.txt"
        file = open(file_name, 'r')

        # csv_data_p = pd.read_csv("./Kmeans_6_13/"+str(digit)+"/test"+str(digit)+"_"+str(csv_file)+".csv_S_p.csv", header=None)
        # csv_data_theta = pd.read_csv("./Kmeans_6_13/"+str(digit)+"/test"+str(digit)+"_"+str(csv_file)+".csv_theta.csv", header=None)
        # csv_data_eta = pd.read_csv("./Kmeans_6_13/"+str(digit)+"/test"+str(digit)+"_"+str(csv_file)+".csv_eta.csv", header=None)

        # file_name = "./Kmeans_6_13/"+str(digit)+"/test"+str(digit)+"_"+str(csv_file)+".csv_B.txt"
        # file = open(file_name, 'r')

        # temp_theta = np.array([])
        # temp_eta = np.array([])
        # for line in file:
        #     parameters = line.strip().split(',')
        #     temp_theta = np.append(temp_theta, float(parameters[3]))
        #     temp_eta = np.append(temp_eta, float(parameters[5]))
        #
        # temp = np.append(temp_theta, temp_eta)
        # if len(temp) < 100:
        #     temp = np.pad(temp, (0, 100 - len(temp)), 'constant', constant_values=0)
        # temp = temp.reshape(1, 100)
        # if flag:
        #     flag = False
        #     idiot = temp
        # else:
        #     idiot = np.append(idiot, temp, axis=0)

        # temp_theta = np.array([])
        # temp_eta = np.array([])
        # for i in range(28):
        #     for j in range(28):
        #         temp_theta = np.append(temp_theta, float(csv_data_theta.iat[i, j]))
        #         temp_eta = np.append(temp_eta, float(csv_data_eta.iat[i, j]))
        # temp = np.append(temp_theta, temp_eta)
        # temp = temp.reshape(1, 784*2)
        # if flag:
        #     flag = False
        #     idiot = temp
        # else:
        #     idiot = np.append(idiot, temp, axis=0)

        # temp_p = np.array([])
        # for i in range(28):
        #     for j in range(28):
        #         temp_p = np.append(temp_p, float(csv_data_p.iat[i, j]))
        # # temp = np.append(temp_theta, temp_eta)
        # temp = temp_p.reshape(1, 784*1)
        # if flag:
        #     flag = False
        #     idiot = temp
        # else:
        #     idiot = np.append(idiot, temp, axis=0)

        DKL = []
        for line in file:
            DKL.append(float(line.strip().split(':')[1]))
        # temp = np.array([DKL[-3], DKL[-2], DKL[-1]]).reshape(1, 3)
        temp = np.array([DKL[-1]]).reshape(1, 1)
        if flag:
            flag = False
            idiot = temp
        else:
            idiot = np.append(idiot, temp, axis=0)

        # count_theta, count_eta = 0, 0
        # count_p = 0
        # sum_p_value = 0
        # sum_theta = 0
        # sum_eta = 0
        # for matrixXd in range(10):
        #     for i in range(28):
        #         for j in range(28):
        #             # if csv_data_theta.iat[28 * matrixXd + i, j] > 0.00000001:
        #             #     count_theta = count_theta + 1
        #             # if csv_data_eta.iat[28 * matrixXd + i, j] > 0.00000001:
        #             #     count_eta = count_eta + 1
        #             # sum_theta += float(csv_data_theta.iat[28 * matrixXd + i, j])
        #             # sum_eta += float(csv_data_eta.iat[28 * matrixXd + i, j])
        #             if csv_data_p.iat[28 * matrixXd + i, j] > 0.00000001:
        #                 count_p = count_p + 1
        #             sum_p_value += float(csv_data_p.iat[28 * matrixXd + i, j])
        # temp = np.array([sum_p_value, count_p]).reshape(1, 2)
        # if flag:
        #     flag = False
        #     idiot = temp
        # else:
        #     idiot = np.append(idiot, temp, axis=0)

        # # 28*28*10 for_Beta
        # sum_theta = 0.0
        # sum_eta = 0.0
        # for line in file:
        #     parameters = line.strip().split(',')
        #     sum_theta += float(parameters[3])
        #     sum_eta += float(parameters[5])
        # temp = np.array([sum_theta, sum_eta]).reshape(1,2)
        # if flag:
        #     flag = False
        #     idiot = temp
        # else:
        #     idiot = np.append(idiot, temp, axis=0)

        # sum_theta = np.array([])
        # sum_eta_org = 0.0
        # sum_eta = np.array([])
        # for line in file:
        #     parameters = line.strip().split(',')
        #     sum_theta = np.append(sum_theta, float(parameters[3]))
        #     sum_eta = np.append(sum_eta, float(parameters[5]))
        #     # sum_theta += float(parameters[3])
        #     # sum_eta += float(parameters[5])
        # temp = np.append(sum_theta, sum_eta)
        # if len(temp) < 1000:
        #     temp = np.pad(temp, (0, 1000 - len(temp)), 'constant', constant_values=0)
        # temp = temp.reshape(1, 1000)
        # if flag == True:
        #     flag = False
        #     idiot = temp
        # else:
        #     idiot = np.append(idiot, temp, axis=0)

# 聚类分析
to_pre = np.array(idiot)
print(to_pre.shape)
X = to_pre
estimator = KMeans(n_clusters=10, max_iter=500, init='random', n_init=15)  # 构造聚类器
estimator.fit(X)  # 聚类
label_pred = estimator.labels_  # 获取聚类标签
print(len(label_pred))

# 出excel统计表
# alll = []
# for i in range(10):
#     this = []
#     for j in range(100):
#         this.append(label_pred[i*100+j])
#     alll.append(this)
# a = pd.DataFrame(alll)
# a = pd.DataFrame(a.values.T)
# result = a.apply(pd.value_counts)
# result.columns = ['Digit0', 'Digit1', 'Digit2', 'Digit3', 'Digit4', 'Digit5', 'Digit6', 'Digit7', 'Digit8', 'Digit9']
# result.index = ['Class0', 'Class1', 'Class2', 'Class3', 'Class4', 'Class5', 'Class6', 'Class7', 'Class8', 'Class9']
# print(result)
# result.fillna(0).to_csv("./aaa/result.csv", header=True)

# 出正常二维坐标值图，非高维
# y = np.array([])
# for ii in range(10):
#     for count in range(100):
#         y = np.append(y, int(ii))
# y = y.astype(int)
# X_tsne = to_pre
# x_min, x_max = X_tsne.min(0), X_tsne.max(0)
# X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
# plt.figure(figsize=(6, 6))
# ax = plt.gca()
# ax.spines['top'].set_visible(False) #去掉上边框
# ax.spines['right'].set_visible(False) #去掉右边框
# ax.spines['bottom'].set_visible(False) #去掉上边框
# ax.spines['left'].set_visible(False) #去掉右边框
# for i in range(X_norm.shape[0]):
#     plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]),
#              fontdict={'weight': 'bold', 'size': 9})
# plt.xticks([])
# plt.yticks([])
# plt.savefig('./aaa/Predicted.png')
# plt.savefig('./aaa/Predicted.svg')
# plt.show()
#
#
# plt.figure(figsize=(6, 6))
# ax = plt.gca()
# ax.spines['top'].set_visible(False) #去掉上边框
# ax.spines['right'].set_visible(False) #去掉右边框
# ax.spines['bottom'].set_visible(False) #去掉上边框
# ax.spines['left'].set_visible(False) #去掉右边框
# for i in range(X_norm.shape[0]):
#     plt.text(X_norm[i, 0], X_norm[i, 1], str(label_pred[i]), color=plt.cm.Set1(label_pred[i]),
#              fontdict={'weight': 'bold', 'size': 9})
# plt.xticks([])
# plt.yticks([])
# plt.savefig('./aaa/Original.png')
# plt.savefig('./aaa/Original.svg')
# plt.show()

# 出t-sne图 放置于aaa文件夹内
# X = idiot
# n_samples, n_features = X.shape
# y = np.array([])
# for ii in range(10):
#     for count in range(100):
#         y = np.append(y, int(ii))
# y = y.astype(int)
#
# tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
# X_tsne = tsne.fit_transform(X)
# x_min, x_max = X_tsne.min(0), X_tsne.max(0)
# X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
# plt.figure(figsize=(8, 8))
# ax = plt.gca()
# ax.spines['top'].set_visible(False) #去掉上边框
# ax.spines['right'].set_visible(False) #去掉右边框
# ax.spines['bottom'].set_visible(False) #去掉上边框
# ax.spines['left'].set_visible(False) #去掉右边框
# for i in range(X_norm.shape[0]):
#     plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]),
#              fontdict={'weight': 'bold', 'size': 9})
# plt.xticks([])
# plt.yticks([])
# plt.savefig('./aaa/Original.png')
# plt.savefig('./aaa/Original.svg')
# plt.show()
#
#
# plt.figure(figsize=(8, 8))
# ax = plt.gca()
# ax.spines['top'].set_visible(False) #去掉上边框
# ax.spines['right'].set_visible(False) #去掉右边框
# ax.spines['bottom'].set_visible(False) #去掉上边框
# ax.spines['left'].set_visible(False) #去掉右边框
#
# for i in range(X_norm.shape[0]):
#     plt.text(X_norm[i, 0], X_norm[i, 1], str(label_pred[i]), color=plt.cm.Set1(label_pred[i]),
#              fontdict={'weight': 'bold', 'size': 9})
# plt.xticks([])
# plt.yticks([])
# plt.savefig('./aaa/Predicted.png')
# plt.savefig('./aaa/Predicted.svg')
# plt.show()

# 两个主要指标计算
y = np.array([])
for ii in range(10):
    for count in range(100):
        y = np.append(y, int(ii))
y = y.astype(int)
label_true = y
filetmp = open('./aaa/indicator.txt', 'w')
filetmp.writelines("AMI:"+str(metrics.adjusted_mutual_info_score(label_true, label_pred))+"\n")
filetmp.writelines("NMI:"+str(metrics.normalized_mutual_info_score(label_true, label_pred))+"\n")
filetmp.writelines("ARI:"+str(metrics.adjusted_rand_score(label_true, label_pred))+"\n")
filetmp.writelines("F1_Micro:"+str(metrics.f1_score(label_true, label_pred, average='micro'))+"\n")
filetmp.writelines("F1_Macro:"+str(metrics.f1_score(label_true, label_pred, average='macro'))+"\n")
filetmp.writelines("AC:"+str(metrics.accuracy_score(label_true, label_pred))+"\n")
filetmp.close()
