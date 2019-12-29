from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName, delim = '\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    dataArr = [list(map(float, line)) for line in stringArr]
    return mat(dataArr)

def pca(dataMat, topNfeat = 9999999):
    # 计算平均值
    meanVals = mean(dataMat, axis=0)
    # 减去平均值
    meanRemoved = dataMat - meanVals
    # 计算协方差矩阵
    covMat = cov(meanRemoved, rowvar=0)
    # 计算特征值和特征向量
    eigVals,eigVects = linalg.eig(mat(covMat))
    # argsort函数返回的是数组值从小到大的索引值
    eigValInd = argsort(eigVals)
    eigValInd = eigValInd[: -(topNfeat + 1) : -1]
    redEigVects = eigVects[:, eigValInd]
    lowDDataMat = meanRemoved * redEigVects
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat

# 测试
# dataMat = loadDataSet('testSet.txt')
# lowDMat, reconMat = pca(dataMat, 2)
# print(shape(lowDMat))
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(dataMat[:,0].flatten().A[0], dataMat[:,1].flatten().A[0], marker='^', s=90)
# ax.scatter(reconMat[:,0].flatten().A[0], reconMat[:,1].flatten().A[0], marker='o',s=50,c='red')
# plt.show()


# 将NaN替换成平均值的函数
def replaceNanWithMean():
    datMat = loadDataSet('secom.data', ' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0], i])
        datMat[nonzero(isnan(datMat[:,i].A))[0], i] = meanVal
    return datMat

# 测试
dataMat = replaceNanWithMean()
meanVals = mean(dataMat, axis=0)
meanRemoved = dataMat - meanVals
covMat = cov(meanRemoved, rowvar=0)
eigVals, eigVects = linalg.eig(mat(covMat))
print(eigVals)