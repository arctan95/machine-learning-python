from numpy import *
import matplotlib.pyplot as plt

# Logistic回归梯度上升优化算法
def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        # 将x0设为1.0
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def sigmoid(inX):
    # 运行时会报错 RuntimeWarning: overflow encountered in exp:   return 1.0 / (1 + exp(-inX))
    # 需要优化sigmoid函数
    # 使用numpy中的tanh函数（双曲正切函数） tanh(x) = [exp(x) - exp(-x)] / [exp(x) - exp(-x)]
    # 0.5 * (1 + tanh(0.5 * inX))经化简可得到原始形式 1.0 / (1 + exp(-inX))
    return 0.5 * (1 + tanh(0.5 * inX))

def gradAscent(dataMatIn, classLabels):
    # 转换为numpy矩阵数据类型
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)
    alpha = 0.001
    maxCycle = 500
    weights = ones((n, 1))
    for k in range(maxCycle):
        h = sigmoid(dataMatrix * weights)
        # 此处计算真实类别和预测类别的差值
        # 对logistic回归函数的对数释然函数的参数项求偏导
        error = (labelMat - h)
        # 更新权值参数
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

# # 测试
# dataArr, labelMat = loadDataSet()
# print(gradAscent(dataArr, labelMat))

# 画出数据集和Logistic回归最佳拟合直线的函数
def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1]); ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1]); ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s = 30, c = 'red', marker = 's')
    ax.scatter(xcord2, ycord2, s = 30, c = 'green')
    # x为numpy.arange格式，并且以0.1为步长从-3.0到3.0切分
    x = arange(-3.0, 3.0, 0.1)
    # 拟合曲线为0 = w0*x0+w1*x1+w2*x2, 故x2 = (-w0*x0-w1*x1)/w2,
    # x0为1,x1为x, x2为y,故有y = (-weights[0] - weights[1] * x) / weights[2]
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1');plt.ylabel('X2');
    plt.show()

# # 测试
# dataArr, labelMat = loadDataSet()
# weights = gradAscent(dataArr, labelMat)
# # x为array格式，weights为matrix格式，故需要调用getA()方法，其将matrix()格式矩阵转为array()格式
# plotBestFit(weights.getA())

# 随机梯度上升算法
def stocGradAscent0(dataMatrix, classLabels):
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

# # 测试
# dataArr, labelMat = loadDataSet()
# weights = stocGradAscent0(array(dataArr), labelMat)
# plotBestFit(weights)

# 改进的随机梯度上升算法
def stocGradAscent1(dataMatrix, classLabels, numIter = 150):
    m, n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
            dataIndex = list(range(m))
            for i in range(m):
                alpha = 4 / (1.0 + j + i) + 0.0001
                randIndex = int(random.uniform(0, len(dataIndex)))
                h = sigmoid(sum(dataMatrix[randIndex] * weights))
                error = classLabels[randIndex] - h
                weights = weights + alpha * error * dataMatrix[randIndex]
                del(dataIndex[randIndex])
    return weights

# # 测试
# dataArr, labelMat = loadDataSet()
# weights = stocGradAscent1(array(dataArr), labelMat, 500)
# plotBestFit(weights)


#Logistic回归分类函数
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt');
    frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)
    errorCount = 0;
    numTestVec = 0.0

    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate

def multiTest():
    numTests = 10; errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum / float(numTests)))

# 测试
multiTest()