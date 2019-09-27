from numpy import *
import feedparser

#词表到向量的转换函数
def loadDataSet():
    # 词条切分后的文档集合，列表每一行代表一个文档，共有6个文档（样本）
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 由人工标注的每篇文档的类标签
    classVec = [0,1,0,1,0,1] # 1代表侮辱性文字 0代表正常言论
    return postingList, classVec

#统计所有文档中出现的词条列表
def createVocabList(dataSet):
    vocabSet = set([])
    # 将文档列表转为集合的形式，保证每个词条的唯一性
    # 然后与vocabSet取并集，向vocabSet中添加没有出现
    # 的新的词条
    for document in dataSet:
        vocabSet = vocabSet | set(document) # 创建两个集合的并集
    return list(vocabSet)


#根据词条列表中的词条是否在文档中出现(出现1，未出现0)，将文档转化为词条向量
def setOfWords2Vec(vocabList, inputSet):
    # 新建一个长度为vocabSet的列表，并且各维度元素初始化为0
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            # 通过列表获取当前word的索引(下标)
            # 将词条向量中的对应下标的项由0改为1
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return  returnVec

# #测试
# listOPosts, listClasses = loadDataSet()
# myVocabList = createVocabList(listOPosts)
# print(myVocabList)
# print(setOfWords2Vec(myVocabList, listOPosts[0]))

#朴素贝叶斯分类器训练函数
# 计算先验概率
# trainMatrix: 词向量矩阵
# trainCategory: 每一个词向量的类别
# 返回每一个单词属于侮辱性和非侮辱性词汇的先验概率, 以及训练集包含侮辱性文档的概率
def trainNB0(trainMatrix, trainCategory):
    # 获取训练样本数量
    numTrainDocs = len(trainMatrix)
    # 获取整个样本中有多少个单词
    numWords = len(trainMatrix[0])
    # sum(trainCategory)就是求贬义的样本个数
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # 词的出现数初始化为1，并将分母初始化为2
    # 这种做法就叫做拉普拉斯平滑(Laplace Smoothing)
    # (参考《统计学习方法》p.64 公式4.11，此处取lambda = 1，分类数 K = 2)
    p0Num = ones(numWords); p1Num = ones(numWords)
    p0Denom = 2.0; p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i] # y=1条件下，统计某个单词出现的个数
            p1Denom += 1  # 累计y=1的所有样本数量
        else:
            p0Num += trainMatrix[i]
            p0Denom += 1
    # 计算先验概率
    # log 防止多个小数相乘出现下溢.
    p1Vect = log(p1Num / p1Denom) # 每个词在该类别中出现的概率
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive

# #测试
# listOPosts, listClasses = loadDataSet()
# myVocabList = createVocabList(listOPosts)
# trainMat = []
# for postinDoc in listOPosts:
#     trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
# p0V, p1V, pAb = trainNB0(trainMat, listClasses)
# print(pAb)
# print(p0V)
# print(p1V)

# 朴素贝叶斯分类函数
# vec2Classify: 测试样本的词向量
# p0Vec: P(x0|Y=0) P(x1|Y=0) P(xn|Y=0)
# p1Vec: P(x0|Y=1) P(x1|Y=1) P(xn|Y=1)
# pClass1: P(=1)Y
# log[P(x1|1)*P(x2|1)*P(x3|1)P(1)]=log[P(x1|1))+log(P(x2|1))+log(P(1)] 对数运算法则
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1) # 对应元素相乘 参考《统计学习方法》p.61 公式4.7
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))

# #测试
# testingNB()

#朴素贝叶斯词袋模型
#每遇到一个单词，词向量中对应值加1
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

#文本解析函数
def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W+', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

#垃圾邮件测试函数
def spamTest():
    docList=[]; classList = []; fullText =[]
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50)); testSet=[]
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet))) # random.uniform(x,y) 随机生成下一个实数,它在 [x, y] 范围内
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print("classification error", docList[docIndex])
    print('the error rate is: ', float(errorCount)/len(testSet))

# #测试
# spamTest()

#高频词去除函数
def calcMostFreq(vocabList,fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token]=fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]

#RSS源分类器函数
def localWords(feed1,feed0):
    import feedparser
    docList=[]; classList = []; fullText =[]
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    top30Words = calcMostFreq(vocabList,fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    trainingSet = list(range(2*minLen)); testSet=[]
    for i in range(20):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[]; trainClasses = []

    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0

    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is: ',float(errorCount)/len(testSet))
    return vocabList,p0V,p1V

# #测试
# ny = feedparser.parse('http://www.nasa.gov/rss/dyn/image_of_the_day.rss')
# sf = feedparser.parse('http://sports.yahoo.com/nba/teams/hou/rss.xml')
# localWords(ny, sf)

#最具表征性的词汇显示函数
def getTopWords(ny,sf):
    import operator
    vocabList,p0V,p1V=localWords(ny,sf)
    topNY=[]; topSF=[]
    for i in range(len(p0V)):
        if p0V[i] > -6.0 : topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -6.0 : topNY.append((vocabList[i],p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        print(item[0])

# #测试
# ny = feedparser.parse('https://newyork.craigslist.org/search/stp/index.rss')
# sf = feedparser.parse('https://sfbay.craigslist.org/search/stp/index.rss')
# getTopWords(ny,sf)



