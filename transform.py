import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import pandas as pd
import csv
import cv2 as cv


def ImageToMatrix(filename):
    im = Image.open(filename)
    width, height = im.size
    im = im.convert("L")
    # print(im.size)
    data = im.getdata()
    data = np.matrix(data,dtype='float')
    new_data = np.reshape(data,(height,width))
    return new_data


def MatrixToImage(data):
    data = data*255
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im


for xxs in range(0, 10):
    count = 0
    count_max = 0
    alll = []
    for root, dirs, files in os.walk('./train/'+str(xxs)+'/'):
        for file in files:
            tmp = ImageToMatrix('./train/'+str(xxs)+'/'+file)
            for items in tmp:
                alll.append(items[0].tolist())
            count = count + 1
            # if count % 10 == 0:
            if 0 == 0:
                matrix1 = []
                for items in alll:
                    matrix1.append(items[0])
                with open('./test-6-13/'+str(xxs)+'/test'+str(xxs)+'_'+str(count)+'.csv', 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    for row in matrix1:
                        writer.writerow(row)
                alll = []

# tt = pd.DataFrame(data=alll)
# tt.to_csv('./testforall.csv')
# filename = './image/1023.png'
# data = ImageToMatrix(filename)
# new_im = MatrixToImage(data)
# plt.imshow(data, cmap=plt.cm.gray, interpolation='nearest')
# new_im.show()


# filename = './image/17.png'
# data = ImageToMatrix(filename)
# print(data)


