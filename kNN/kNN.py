from numpy import *
from os import listdir
import operator

#创建简单样本集
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

#k-近邻算法
#inX:用于分类的输入向量 dataSet:训练样本集 labels:标签向量 k:选择最近邻居的数目
def classify0(inX, dataSet, labels, k):
    #shape[0]返回行数，shape[1]返回列数
    dataSetSize = dataSet.shape[0]
    #tile()将原矩阵横向、纵向地复制
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    #axis=1表示按行相加 , axis=0表示按列相加
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances ** 0.5
    #argsort()返回数组值从小到大的索引值
    sortedDistIndices = sqDistances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    #按照classCount第二个元素即次数逆序排序
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

# #测试，结果为B
# group, labels = createDataSet()
# print(classify0([0,0], group,labels, 3))


#将文本记录转换为NumPy的解析程序
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOflines = len(arrayOLines)
    returnMat = zeros((numberOflines, 3)) #维度3也可以更改为其他值以适应需求
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t') # \t表示空四个字符,也称缩进
        returnMat[index, :] = listFromLine[0 : 3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

# #测试
# datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
# print(datingDataMat, datingLabels[0: 20])
#
# #使用Matplotlib创建散点图
# import matplotlib
# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
# plt.show()

#归一化特征值
def autoNorm(dataSet):
    minVals = dataSet.min(0) #0表示每列最小值，那么1表示每行最小值
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))#shape同时返回行数和列数
    m = dataSet.shape[0] #返回行数
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet  / tile(ranges, (m, 1))#特征值相除，不是矩阵除法
    return normDataSet, ranges, minVals

# #测试
# normMat, ranges, minVals = autoNorm(datingDataMat)
# print(normMat)

#分类器针对约会网站的测试代码
def datingClassTest():
    hoRatio = 0.10 #设置用于测试的样本比例
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs : m, :], datingLabels[numTestVecs : m], 3)
        print("\nthe classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("\nthe total error rate is: %f" % (errorCount / float(numTestVecs)))

#约会网站预测函数
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print("You will probably like this person:",resultList[classifierResult - 1])

# #测试
# classifyPerson()

#将图像转换为向量
def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline();
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect

# #测试
# testVector = img2vector('testDigits/0_13.txt')
# print(testVector[0, 0: 31])

#手写数字识别系统
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)

    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)

        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d,the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1.0
        print("\nthe total number of errors is: %d" % errorCount)
        print("\nthe total error rate is: %f" % (errorCount / float(mTest)))

#测试
handwritingClassTest()