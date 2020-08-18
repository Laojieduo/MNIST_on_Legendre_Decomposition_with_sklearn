import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets


idiot = np.array([[]])
flag = True
for digit in range(10):
    for csv_file in range(1, 101):
        file_name = "./Kmeans_6_13/"+str(digit)+"/test"+str(digit)+"_"+str(csv_file)+".csv_B.txt"
        file = open(file_name, 'r')
        sum_theta = np.array([])
        sum_eta = np.array([])
        for line in file:
            parameters = line.strip().split(',')
            sum_theta = np.append(sum_theta, float(parameters[3]))
            sum_eta = np.append(sum_eta, float(parameters[5]))
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
X = idiot
n_samples, n_features = X.shape
y = np.array([])
for ii in range(10):
    for count in range(100):
        y = np.append(y, int(ii))
y = y.astype(int)
print(y)


tsne = manifold.TSNE(n_components=3, init='pca', random_state=501)
X_tsne = tsne.fit_transform(X)
print(X_tsne)
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
print(X_norm)
print(X_norm.shape)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(X_norm.shape[0]):
    ax.text(X_norm[i, 0], X_norm[i, 1], X_norm[i, 2], str(y[i]), color=plt.cm.Set1(y[i]))
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
# plt.figure(figsize=(8, 8, 8))
# for i in range(X_norm.shape[0]):
#     plt.text(X_norm[i, 0], X_norm[i, 1], X_norm[i, 2], str(y[i]), color=plt.cm.Set1(y[i]),
#              fontdict={'weight': 'bold', 'size': 9})
# plt.xticks([])
# plt.yticks([])
# plt.zticks([])
plt.show()
