# -*-coding:utf-8 -*-

__author__ = 'sima'

from matplotlib import pyplot as plt
from ch6 import svmMLiA

dataArr, labelArr = svmMLiA.loadDataSet('testSet.txt')
posX = []; posY = []
negX = []; negY = []
for i in range(len(labelArr)):
    if labelArr[i] > 0:
        posX.append(dataArr[i][0])
        posY.append(dataArr[i][1])
    else:
        negX.append(dataArr[i][0])
        negY.append(dataArr[i][1])

# 支持向量
b, alphas = svmMLiA.smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
svX = []; svY = []
for i in range(len(labelArr)):
    if alphas[i] > 0.0:
        svX.append(dataArr[i][0])
        svY.append(dataArr[i][1])

plt.xlabel('x')
plt.ylabel('y')
plt.scatter(posX, posY, marker='s', c='red', label='pos')
plt.scatter(negX, negY, marker='o', c='green', label='neg')
plt.scatter(svX, svY, c='', edgecolors='b', marker='o', s=100)  # 标注支持向量 edgecolors是控制圆圈的边缘颜色，c是控制圆心的颜色
plt.legend(loc='best')
plt.show()
