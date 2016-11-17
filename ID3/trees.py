# coding:utf-8
import operator
from math import log
import treePlotter as tp


def calcEnt(dataSet):
    numEnts = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    ent = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEnts
        ent -= prob * log(prob, 2)
    return ent


def loadData():
    f = open('dataSet.txt')
    dataSet = [inst.strip().split(',') for inst in f.readlines()]
    f = open('labels.txt')
    labels = [la.strip().decode("utf-8") for la in f.readlines()]
    return dataSet, labels


# axis:要分类的特征所在列，value:划分属性
def splitDS(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEnt = calcEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEnt = 0.0
        for value in uniqueVals:
            subDataSet = splitDS(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEnt += prob * calcEnt(subDataSet)
        infoGain = baseEnt - newEnt
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDS(dataSet, bestFeat, value), subLabels)
    return myTree


def main():
    dataSet, labels = loadData()
    print(dataSet)
    print(labels)
    tree = createTree(dataSet, labels)
    print(tree)
    tp.createPlot(tree)


if __name__ == '__main__':
    main()
