from  numpy import *
import operator #运算符模块
def createDataSet():
     group = array([[1.0,1.1], [1.0,1.0], [0,0], [0, 0.1]])
     labels = ['A', 'A', 'B', 'B']
     return group, labels

group, labels = createDataSet()
print(group)
print(labels)

#inx: 用于分类的输入向量
#dataSet: 输入的训练样本集
#labels: 标签向量
#k：用于选择最近邻居的数目
def classify0(inx, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    #tile(a, (m, n)) : 构造 m * n 个a
    diffMat = tile(inx, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistindicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteILabel = labels[sortedDistindicies[i]]
        #get(key, default = None) 
        #函数返回指定键的值，如果值不在字典中返回默认值
        #key -- 字典中要查找的键。
        #default -- 如果指定键的值不存在时，返回该默认值值。 
        classCount[voteILabel] = classCount.get(voteILabel, 0) + 1
    #Python 3 renamed dict.iteritems -> dict.items
    #dict.items()将dict的元素对按照list返回
    #key = operator.itemgetter(1)是一个函数，表示取classCount.items()的第1个域。
    #这句的意思：对classCount的值进行排序
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

print(classify0([0,0], group, labels, 3))

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    #创建一个numberOfLines * 3大小的值0的矩阵
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip() #截取掉所有的回车字符
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
print(datingDataMat)
print(datingLabels)

import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
#ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 15.0*array(datingLabels), 15.0*array(datingLabels))
plt.show()

#归一化特征值
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

normMat, ranges, minVals = autoNorm(datingDataMat)
print(normMat)
print(ranges)
print(minVals)

def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" %(classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
        print("the total error rate is: %f" %(errorCount / float(numTestVecs)))
datingClassTest()

# def classifyPerson():
#     resultList = ['not at all', 'in small doses', 'in large doses']
#     percentTats = float(raw_input("percentage of time spent playing video games?"))
#     ffMiles = float(raw_input("frequent flier miles earned per year?"))
#     iceCream = float(raw_input("liters of ice cream consumed per year?"))
#     datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
#     normMat, ranges, minVals = autoNorm(datingDataMat)
#     inArr = array([ffMiles, percentTats, iceCream])
#     classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
#     print("You will probably like this person: ", resultList[classifierResult - 1])

# classifyPerson()