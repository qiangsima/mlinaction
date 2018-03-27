# -*-coding:utf-8 -*-

__author__ = 'sima'

from numpy import *


def loadSimpData():
    datMat = matrix([[1., 2.1],
                     [2., 1.1],
                     [1.3, 1.],
                     [1., 1.],
                     [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels


def stumpClassify(datMat, dimen, threshVal, threshIneq):
    """通过阈值比较对数据进行分类"""
    retArray = ones(shape(datMat)[0], 1)
    if threshIneq == 'lt':
        retArray[datMat[:, dimen] <= threshVal] = -1.0
    else:
        retArray[datMat[:, dimen] > threshVal] = -1.0
    return retArray


def buildStump(datArr, classLabels, D):
    datMat = mat(datArr)
    labelMat = mat(classLabels).T
    m, n = shape(datMat)
    numSteps = 10.0
    bestStump = {}
    bestClassEst = mat(zeros((m, 1)))
    minErr = inf
    for i in range(n):      # 遍历数据集中的每一个特征
        rangeMin = datMat[:, i].min()
        rangeMax = datMat[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps) + 1):      # 遍历每一个步长
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(datMat, i, threshVal, inequal)
                errArr = mat(ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedErr = D.T * errArr      # 计算加权错误率
                if weightedErr < minErr:
                    minErr = weightedErr
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minErr, bestClassEst

def main():
    import plotdemo
    datMat, classLabels = loadSimpData();
    plotdemo.scatter(datMat, classLabels)


if __name__ == '__main__':
    main()
