# coding:utf-8

from numpy import *
from os import listdir
import operator
import matplotlib
import matplotlib.pyplot as plt


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def file2mat(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    minVal = dataSet.min(0)
    maxVal = dataSet.max(0)
    range = maxVal - minVal
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVal, (m, 1))
    normDataSet = normDataSet / tile(range, (m, 1))
    return normDataSet, range, minVal


def main():
    datingmat, datinglabel = file2mat('datingTestSet2.txt')
    norm, r, min = autoNorm(datingmat)
    fig = plt.figure()
    ax1 = plt.subplot(111)
    ax1.scatter(datingmat[:, 0], datingmat[:, 1],
               15.0*array(datinglabel), 10.0*array(datinglabel))
    ax1.set_xlabel(r"每年获取的飞行常客里程数")
    ax1.set_ylabel(r"玩视频游戏所耗时间百分比")

    plt.show()


if __name__ == '__main__':
    main()
