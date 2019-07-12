import numpy as np
import operator
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as kNN


def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        filenameStr = trainingFileList[i]
        classNumber = int(filenameStr.split('_')[0])
        hwLabels.append(classNumber)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % (filenameStr))
    neigh = kNN(n_neighbors=3, algorithm='auto')
    neigh.fit(trainingMat, hwLabels)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        filenameStr = testFileList[i]
        classNumber = int(filenameStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' %(filenameStr))
        classifierResult = neigh.predict(vectorUnderTest)
        print('分类返回结果为%d\t真实值结果为%d' %(classifierResult, classNumber))
        if(classifierResult != classNumber):
            errorCount += 1.0
    print('总错误了%d个数据\n错误率为%f%%' % (errorCount, errorCount / mTest * 100))


if __name__ == '__main__':
    handwritingClassTest();








