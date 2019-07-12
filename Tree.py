from math import log
import operator
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import pickle
from sklearn import tree
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import pydotplus
from sklearn.externals.six import StringIO


# def createDataSet():
#     dataSet = [[0, 0, 0, 0, 'no'],
#             [0, 0, 0, 1, 'no'],
#             [0, 1, 0, 1, 'yes'],
#             [0, 1, 1, 0, 'yes'],
#             [0, 0, 0, 0, 'no'],
#             [1, 0, 0, 0, 'no'],
#             [1, 0, 0, 1, 'no'],
#             [1, 1, 1, 1, 'yes'],
#             [1, 0, 1, 2, 'yes'],
#             [1, 0, 1, 2, 'yes'],
#             [2, 0, 1, 2, 'yes'],
#             [2, 0, 1, 1, 'yes'],
#             [2, 1, 0, 1, 'yes'],
#             [2, 1, 0, 2, 'yes'],
#             [2, 0, 0, 0, 'no']]
#     labels = ['不放贷', '放贷']
#     return dataSet, labels


def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],                        #数据集
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']        #特征标签
    return dataSet, labels


def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVect in dataSet:
        currentLabel = featVect[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        print("第%d个特征的增益为：%.3f" % (i, infoGain))
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    classCounts = {}
    for vote in classList:
        if vote not in classCounts.keys():
            classCounts[vote] = 0
        classCounts[vote] += 1
    sortedClassCount = sorted(classCounts.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels, featLabels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1 or len(labels) == 0:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    featLabels.append(bestFeatLabel)
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featVals = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featVals)
    for value in uniqueVals:
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), labels, featLabels)
    return myTree


def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = next(iter(myTree))
    secondDic = myTree[firstStr]
    for key in secondDic.keys():
        if type(secondDic[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDic[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = next(iter(myTree))
    secondDic = myTree[firstStr]
    for key in secondDic.keys():
        if type(secondDic[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDic[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


def plotNode(nodeText, centerPt, parentPt, nodeType):
    arrow_args = dict(arrowstyle='<-')
    font = FontProperties(fname='simsun.ttc', size=14)
    createPlot.ax1.annotate(nodeText, xy=parentPt, xycoords='axes fraction', xytext=centerPt, textcoords='axes fraction', va="center", ha="center", bbox=nodeType, arrowprops=arrow_args, FontProperties=font)


def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)



def plotTree(myTree, parentPt, nodeTxt):
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")
    leafNode = dict(boxstyle="round4", fc="0.8")
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = next(iter(myTree))
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()


def classify(inputTree, featLabels, testVec):
    firstStr = next(iter(inputTree))
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def storeTree(inputTree, filename):
    with open(filename, 'wb') as fw:
        pickle.dump(inputTree, fw)


def grabTree(filename):
    fr = open(filename, 'rb')
    return pickle.load(fr)


if __name__ == '__main__':
    # dataSet, features = createDataSet()
    # print(dataSet)
    # print(calcShannonEnt(dataSet))
    # print("最优特征索引值：" + str(chooseBestFeatureToSplit(dataSet)))
    # featLabels = []
    # myTree = createTree(dataSet, features, featLabels)
    # testVec = [0, 1]
    # print(myTree)
    # createPlot(myTree)

    # storeTree(myTree, 'classifierStorage.txt')

    # result = classify(myTree, featLabels, testVec)
    # if result == "yes":
    #     print("放贷")
    # if result == 'no':
    #     print("不放贷")

    # tree = grabTree('classifierStorage.txt')
    # print(tree)

    # fr = open('lenses.txt')
    # lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    # print(lenses)
    # lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    # clf = tree.DecisionTreeClassifier()
    # lenses = clf.fit(lenses, lensesLabels)

    with open('lenses.txt', 'r') as fr:
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lenses_target = []
    for each in lenses:
        lenses_target.append([each[-1]])
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lenses_list = []
    lenses_dict = {}
    for each_label in lensesLabels:
        for each in lenses:
            lenses_list.append(each[lensesLabels.index(each_label)])
        lenses_dict[each_label] = lenses_list
        lenses_list = []
    lenses_pd = pd.DataFrame(lenses_dict)
    print(lenses_pd)
    le = LabelEncoder()
    for col in lenses_pd.columns:
        lenses_pd[col] = le.fit_transform(lenses_pd[col])
    # print(lenses_pd)
    clf = tree.DecisionTreeClassifier(max_depth=4)
    clf = clf.fit(lenses_pd.values.tolist(), lenses_target)
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data,  # 绘制决策树
                         feature_names=lenses_pd.keys(),
                         class_names=clf.classes_,
                         filled=True, rounded=True,
                         special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("tree.pdf")









