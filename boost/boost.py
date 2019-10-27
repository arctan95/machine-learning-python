from numpy import *
import matplotlib.pyplot as plt
def loadSimpData():
    datMat = array([[1.,2.1],
                     [2.,1.1],
                     [1.3,1.],
                     [1.,1.],
                     [2.,1.]])
    classLabels = [1.0,1.0,-1.0,-1.0,1.0]
    return datMat, classLabels
dataMat, classLabels = loadSimpData()

# 最佳单层决策树生成函数
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray

def buildStump(dataArr, classLabels, D):
    dataMatrix = mat(dataArr);labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    numSteps = 10.0; bestStump = {}; bestClassEst = mat(zeros((m, 1)))
    minError = inf
    for i in range(n):
        rangeMin = dataMatrix[:, i].min();rangeMax = dataMatrix[:, i].max();
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = mat(ones((m, 1)))
                errArr[predictVals == labelMat] = 0
                weightedError = D.T * errArr
                # print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClassEst

# 测试
# # 初始权重 1/N
# D = mat(ones((5,1)) / 5)
# print(buildStump(dataMat, classLabels, D))

# 基于单层决策树的AdaBoost训练过程
# dataArr 数据集, classLabels 类别标签, numlt = 40 迭代次数（默认40）
def adaBoostTrainDS(dataArr, classLabels, numlt = 40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m,1)) / m)
    # aggClassEst记录每个数据点的类别估计累计值
    aggClassEst = mat(zeros((m, 1)))
    for i in range(numlt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print("D:", D.T)
        # max(error, 1e-16))用于确保在没有错误时不会发生除0溢出
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16))) # 科学计数法 即exp(-16)
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print("classEst:", classEst.T)
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)
        D = multiply(D, exp(expon))
        D = D / D.sum()
        aggClassEst += alpha * classEst
        print("aggClassEst:", aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print("total error:", errorRate, "\n")
        # 训练错误为0时，退出循环
        if errorRate == 0.0: break
    return weakClassArr,aggClassEst
# 测试
# classifierArray = adaBoostTrainDS(dataMat, classLabels, 9)
# print(classifierArray)

# AdaBoost分类函数
def adaClassify(dataToClass, classifierArray):
    dataMatrix = mat(dataToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArray)):
        classEst = stumpClassify(dataMatrix, classifierArray[i]['dim'], classifierArray[i]['thresh'], classifierArray[i]['ineq'])
        aggClassEst += classifierArray[i]['alpha'] * classEst
        print(aggClassEst)
    return sign(aggClassEst)

# 测试
# print(adaClassify([0,0],classifierArray))

# 马疝病的预测

# 自适应数据加载函数
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

# 测试
# dataArr, labelArr = loadDataSet('horseColicTraining2.txt')
# classifierArray = adaBoostTrainDS(dataArr, labelArr, 10)
# print(classifierArray)
# print("-------------------------------------------------")
# testArr, testLabelArr = loadDataSet('horseColicTest2.txt')
# prediction10 = adaClassify(testArr, classifierArray)
# errArr = mat(ones((67,1)))
# print(errArr[prediction10 != mat(testLabelArr).T].sum())

# ROC曲线的绘制及AUC计算函数
def plotROC(predStrengths, classLabels):
    cur = (1.0, 1.0)  # cursor
    ySum = 0.0  # variable to calculate AUC
    numPosClas = sum(array(classLabels) == 1.0)
    yStep = 1 / float(numPosClas);
    xStep = 1 / float(len(classLabels) - numPosClas)
    sortedIndicies = predStrengths.argsort()  # get sorted index, it's reverse
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    # loop through all the values, drawing a line segment at each point
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0;
            delY = yStep;
        else:
            delX = xStep;
            delY = 0;
            ySum += cur[1]
        # draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c='b')
        cur = (cur[0] - delX, cur[1] - delY)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print("the Area Under the Curve is: ", ySum * xStep)

# 测试
dataArr, labelArr = loadDataSet('horseColicTraining2.txt')
classifierArray, aggClassEst = adaBoostTrainDS(dataArr, labelArr, 10)
plotROC(aggClassEst.T, labelArr)