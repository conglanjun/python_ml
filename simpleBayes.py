import numpy as np
from functools import reduce

def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],  # 切分的词条
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 类别标签向量，1代表侮辱性词汇，0代表不是
    return postingList, classVec


def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec


def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = np.zeros(numWords)
    p1Num = np.zeros(numWords)
    p0Denom = 0.0
    p1Denom = 0.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = p1Num / p1Denom
    p0Vect = p0Num / p0Denom
    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = reduce(lambda x,y:x*y, vec2Classify * p1Vec) * pClass1
    p0 = reduce(lambda x,y:x*y, vec2Classify * p0Vec) * (1.0 - pClass1)
    print('p0:', p0)
    print('p1:', p1)
    if p1 >p0:
        return 1
    else:
        return 0


def testingNB():
    listOPosts, listClasses  =loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []



if __name__ == '__main__':
    postingLIst, classVec = loadDataSet()
    print('postingList:\n', postingLIst)
    myVocabList = createVocabList(postingLIst)
    print("myVocabList:\n", myVocabList)
    trainMat = []
    for postinDoc in postingLIst:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    # print("trainMat:\n", trainMat)
    p0V, p1V, pAb = trainNB0(trainMat, classVec)
    print("p0V:\n", p0V)
    print("p1V:\n", p1V)
    print("classVec:\n", classVec)
    print("pAb:\n", pAb)





