from numpy import *
import operator

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

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

#测试，结果为B
group, labels = createDataSet()
print(classify0([0,0], group,labels, 3))
