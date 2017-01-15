# k-Nearest Neighbors classification algorithm
# 给定一个数据集，对于新的输入实例，在数据集中找到与该实例最邻近的k个实例，这k个实例中大部分所属的类，即为新实例分属的类

import numpy as np
import operator

# 构造数据集
def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])  # 4x2 二维数值
    labels = ['A','A','B','B']
    return group,labels


#
# pseudocode:
# for every point in our dataset:                        遍历数据集中的每个点：
#    calculate the distance between inX and current point           计算新输入实例inX与每个点的距离
#    sort the distances in increasing order                         按距离升序排列
#    take k items with lowest distances to inX                      取前k个最近邻的点
#    find the majority class among these items                      找到这些点大多数所属的分类（频率）
#    return the majority class as our prediction for the class of inX   返回这个占大多数的类，即为新实例的类别


def classify0(inX,dataSet,labels,k):  # 四个参数：新的输入向量inX，数据集，标签向量y，k
    #print('k=',k)
    dataSetSize = dataSet.shape[0]    # 数据集样本量
    # 计算欧氏距离： d = √ (x0-x1)²+(x'0-x'1)²+(x"0-x"2)²...
    diffMat = np.tile(inX,(dataSetSize,1)) - dataSet  # np.tile张量积，inX"铺砖"成数据集矩阵形状，矩阵减法
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()  # np.argsort 返回排序后值的索引（列表）。根据索引取得labels。最终要的并不是距离。
    # 取前k个点的分类（标签）
    classCount = {}
    for i in range(k):
        #print('第 %d 邻近的点：' % (i+1))
        voteIlabel = labels[sortedDistIndicies[i]]  # 返回具体的标签值
        #print('分类标签是：',voteIlabel)
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1 #统计各标签出现的频率。dict.get取得键(标签）对应的值(频率)。
        #print('至此各分类标签出现的频率（字典形式）：',classCount)
        sortedClassCount = sorted(classCount.items(),  # python2中dict.items()返回[(键值对的元组)列表]，py3中要加list()
                 key = operator.itemgetter(1),reverse=True)  # 按照第二个元素即频率，从大到小排序。
                                                                # sorted(iterable，cmp，key，reverse）key为函数
        #print('投票：',sortedClassCount)
        #print('--------------')
    return sortedClassCount[0][0]   # 返回最终分类结果

# 测试：
# group,labels = createDataSet()
# print('数据集：\n',group,'\n',labels)
# print('--------------')
# print('测试：')
# print('k=3，预测[0,0]的分类为：',classify0([0,0],group,labels,3))


#----------- 示例：使用KNN改进约会网站配对效果-----------------------------
# 3个特征：飞行里程、娱乐时间占比、冰淇淋消费量

# 解析文本数据，将文本转换成分类器可用的格式：
def file2matrix(filename):
    fr = open(filename)         # 打开文件
    arrayOLines = fr.readlines() # 读取文本所有行
    numberOfLines = len(arrayOLines) #得到文本行数
    returnMat = np.zeros((numberOfLines,3))  #创建以0填充的矩阵，3为特征数
    classLabelVector = []
    index = 0
    for line in arrayOLines:  # 逐行处理
        line = line.strip()   # 格式化
        listFromLine =  line.split('\t') # 按'\t'分割成元素列表
        returnMat[index,:] = listFromLine[0:3] # 取前3个元素填充原来的0矩阵
        classLabelVector.append(int(listFromLine[-1])) # 最后一个元素为标签，得到标签列表
        index += 1
    return returnMat,classLabelVector  # 返回数据矩阵、标签列表

datingDataMat,datingLabels = file2matrix('datingTestSet.txt')
# print(datingDataMat)
# print(datingLabels[:20])


# ----分析数据：使用Matplotlib 创建散点图-----------
# import matplotlib
# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(datingDataMat[:,1],datingDataMat[:,2],   # 用矩阵的第二、三列作图
#            15.0*np.array(datingLabels),15.0*np.array(datingLabels)) # 根据标签用不同颜色区分散点
# plt.show()

# 另一种更高效的方法：用pandas 处理----------
# import pandas as pd
# df=pd.read_csv('datingTestSet.txt',,sep='\t',index_col=False,names=['Fly','gametime','ice','label'])
# df.plot(kind='scatter',x='Fly',y='ice',c='label',colormap='spring')  #必须先设置c，才能设置colormap
# plt.show()


#---- 准备数据：归一化数值
# 若三个特征同等重要，等权值，但是它们的取值范围不同，则需要归一化处理，将取值范围转换到(0,1):
# newValue = (oldValue-min)/(max-min)
def auotNorm(dataSet):   # np.array
    minVals = dataSet.min(axis=0) # 参数0使得从各列中选取最小值，返回最小值组成的array(列表)，长度等于列数。参数1沿行，长度=行数。
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals,(m,1))
    normDataSet = normDataSet/np.tile(ranges,(m,1)) # 此处是具体特征值相除，而非矩阵除法。numpy 中linalg.sove(matA,matB)为矩阵除法
    return normDataSet,ranges,minVals

#测试
# normMat,ranges,minVals = auotNorm(datingDataMat)
# print(normMat)
# print(ranges)
# print(minVals)

# --------测试分类器性能：错误率检测--------------------------
# 整个已知数据集中，随机90%作训练集，10%作测试集
# 错误率 = 犯错次数 / 测试集总数

def datingClassTest():
    hoRatio = 0.10  #  测试集占比
    datingDataMat,datingLabels=file2matrix('datingTestSet.txt') # 得到数据集
    normMat,ranges,minVals = auotNorm(datingDataMat)    # 归一化处理
    m = normMat.shape[0]                                        # 数据集总量
    numTestVecs = int(m*hoRatio)                            # 测试集总量
    errorCount = 0.0
    for i in range(numTestVecs):    # 遍历测试集，得到每个测试样本的分类结果
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],
                                     datingLabels[numTestVecs:m],3)  # k=3
        print('the classifier came back with: %d,the real answer is: %d' % (classifierResult,datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1
    print('the total error rate is: %f' % (errorCount / float(numTestVecs)))

# 测试：
# datingClassTest()

# -------构建完整预测系统-----------------------------

def classifyPerson():
    resultList = ['not at all','in small doses','in large doses']
    percentTat = float(input('percentage of time spent playing video games?')) #py3中raw_input()被重命名为input
    ffMiles = float(input('frequent flier miles earned per year?'))
    icecream = float(input('liters of ice cream consumed per year?'))
    datingDataMat,datingLabels = file2matrix('datingTestSet.txt')
    normMat,ranges,minVals=auotNorm(datingDataMat)
    inArr = np.array([ffMiles,percentTat,icecream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print('You will probably like this person: ',resultList[classifierResult-1])

# 测试执行：
# classifyPerson()

# ----------------------------示例二：手写数字识别---------------------------------------

# 为了使用前面的分类器，必须将图像格式化处理为一个向量。
# ----将32*32的二进制图像矩阵 转换为 1*1024的向量：
def img2vector(filename):
    returnVect = np.zeros((1,1024)) # 创建 1*1024 的array数组(向量）
    fr = open(filename)             # 打开文档
    for i in range(32):             # 循环读取文档前32行
        lineStr = fr.readline()     # 读取32行中的第i行，每行长度为32
        for j in range(32):         # 将每行的前32个字符存储到np数组中
            returnVect[0, 32 * i + j] = int(lineStr[j])  # 填充数组中坐标(0, 32*i+j)的值
    return returnVect

# ----测试算法：使用k-NN识别手写数字：
from os import listdir    # listdir() 给定目录下的文件名 列表。因为一个数字样本单独一个文档，有上百个数字样本。
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits') # 获取训练集目录文件下的所有数字样本文档。str列表
    m = len(trainingFileList)                   # 样本数
    trainingMat = np.zeros((m,1024))            # 转换成 (m行 x 1024列)，每行代表一个数字图像样本
    # 从文件名解析分类数字，如'0_1.txt'、'1_10.txt'：
    for i in range(m):                          # 遍历每行（每个数字）
        fileNameStr = trainingFileList[i]       # 文件名，str
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0]) # 得到该数字样本的分类
        hwLabels.append(classNumStr)            # 分类标签集y
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr) # 填充训练数据集的每行向量
     # 对测试集做类似操作，不同的是并不将各文件载入矩阵，而是使用classify0测试。
    testFileList = listdir('testDigits') #获取测试集目录下的文件
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print('the classifier came back with: %d, the real answer is: %d' % (classifierResult,classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1
    print('\nthe total number of errors is: %d' % errorCount)
    print('\nthe total error rate is: %f' % (errorCount/float(mTest)))

# 测试：
#handwritingClassTest()

