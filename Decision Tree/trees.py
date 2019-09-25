from math import log
import operator
from treePlotter import retrieveTree,createPlot

#计算给定数据集的香农熵
def calcShannonEnt(dataSet) :
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

#创建简单样本集
def createDataSet():
    dateSet = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels = ['no surfacing','flippers']
    return dateSet, labels

# #测试
# myDat,labels = createDataSet()
# print(myDat)
# print(calcShannonEnt(myDat))

#按照给定特征划分数据集
#dataSet:待划分的数据集 axis:划分数据集的特征 value:需要返回的特征的值
def splitDataSet(dataSet, axis, value):
    #python是引用传递，因此新建列表对象以防止影响原始数据集
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reduceFeatVec)
    return retDataSet

# #测试
# myDat,labels = createDataSet()
# print(myDat)
# print(splitDataSet(myDat, 0, 1))
# print(splitDataSet(myDat, 0, 0))

#选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if(infoGain > bestInfoGain):
            bestInfoGain =infoGain
            bestFeature = i
    return bestFeature

# #测试
# myDat,labels = createDataSet()
# print(chooseBestFeatureToSplit(myDat))
# print(myDat)

#多数表决
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    return  sortedClassCount[0][0]

#创建树的函数代码
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    #停止条件1：所有的类标签完全相同，得到叶子节点
    #count()返统计次数
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    #停止条件2：使用完所有特征后，仍不能很好分组，此时采用多数表决法
    if len(dataSet[0]) == 1:
        return  majorityCnt(classList)

    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        #复制类标签，防止引用传递影响原始列表
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

# #测试
# myDat,labels = createDataSet()
# myTree = createTree(myDat,labels)
# print(myTree)


#使用决策树的分类函数
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    # 使用index方法查找当前列表第一个匹配firstStr变量的元素
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    # 函数来判断一个对象是否是一个已知的类型，类似 type()
    # type() 不会认为子类是一种父类类型，不考虑继承关系
    # isinstance() 会认为子类是一种父类类型，考虑继承关系
    if isinstance(secondDict[key], dict):
        classLabel = classify(secondDict[key], featLabels, testVec)
    else:
        classLabel = secondDict[key]
    return  classLabel

# #测试
# myDat,labels = createDataSet()
# print(labels)
# myTree = retrieveTree(0)
# print(myTree)
# print(classify(myTree, labels, [1, 0]))
# print(classify(myTree, labels, [1, 1]))

#使用pickle模块存储决策树
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'wb+')
    pickle.dump(inputTree, fw)
    fw.close

def grabTree(filename):
    import pickle
    fr = open(filename,'rb')
    return pickle.load(fr)

# #测试
# myTree = retrieveTree(0)
# storeTree(myTree,'classifierStorage.txt')
# print(grabTree('classifierStorage.txt'))

#使用决策树预测隐形眼镜类型
fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age','prescript','astigmatic','tearRate']
lensesTree = createTree(lenses, lensesLabels)
print(lensesTree)
createPlot(lensesTree)

