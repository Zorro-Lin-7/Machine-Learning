# Pseudo-code for a function called creataBranch():
# Check if every item in the dataset is in the same class:
# 	if so :
# 		return the class label
# 	else:
# 		find the best feature to split the data
# 		split the dataset
# 		create a branch node
# 			for each split
# 				call createaBranch and add the result to the branch node
# 		return branch node

# 有的决策树采用二分法划分数据集，但这里依据属性划分。如果属性划分出4个可能值，就按此切成4块，创建4个分支。
# 这里采用ID3算法。ID3可以划分标称型数据集，但无法直接处理数值型数据。另有最流行的CART算法
# 每次划分数据集只选取一个特征属性。若有20个特征，第一个拿谁开刀？
# -----------信息增益---------------------------------------------
# 划分数据集的的大原则是：将无序的数据变得更加有序。
# 信息增益：数据集划分前后的信息变化，用熵度量；另一个度量方式是基尼不纯度。
# 计算每个特征划分的信息增益，选择使信息增益最高的特征
# 随机变量X 的熵定义： H(X) = - ∑ p(x_i)·log p(x_i） 。以 2或e 为底, x_i为分类类别
# 熵只依赖X的分布，与X取值无关


import operator


# ---------计算给定数据集的熵-------------------------------------

# 简单的数据集生成函数：
def createDataSet():
    dataSet = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataSet,labels

from math import log
def calcShannonEnt(dataSet):
    numentries = len(dataSet)  # 统计样本数量
    labelCounts = {}           #为所有可能分类创建字典
    for featVec in dataSet:     # 遍历数据集中的每个样本（特征向量）
        currentLabel = featVec[-1] # 数据集中最后一列的数值 作为键值
        if currentLabel not in labelCounts.keys(): # 如果当前键值（类别标签）不存在
            labelCounts[currentLabel] = 0         # 扩展字典并将当前键值加入字典
        labelCounts[currentLabel] += 1    # 统计当前类别出现的频数
    shannonEnt = 0.0
    for key in labelCounts:  # 遍历所有类标签
        prob = float(labelCounts[key])/numentries  # 使用类标签出现的频率作概率
        shannonEnt -= prob * log(prob,2)           # 计算整个数据集的熵：加和所有子集的熵
    return shannonEnt


#测试：
# myDat,labels = createDataSet()
# print(calcShannonEnt(myDat))
# myDat[0][-1]='maybe'      #熵越高，则数据越复杂无序。添加更多分类进行测试
# print(calcShannonEnt(myDat))

# -----------按照给定的特征划分数据集---------------------------------------
# 想象特征数为2的二维散点图，沿哪个特征（x还是y轴）切割？

def splitDataSet(dataSet,axis,value): # 三个参数：带划分的数据集、用于划分的特征、特征的返回值
    retDataSet = []   # 该函数将在同一数据集上被调用多次，为了不修改原始数据集，创建一个新的列表对象
    for featVec in dataSet:  # 数据集是个嵌套列表，遍历列表里的每个元素（特征向量，列表）
        if featVec[axis] == value:  # 若[特征向量]某维=value，以此切割。可以是数值，也可以是字符串。因为只做是非判断，不做计算
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])   # extend与append相似，但返回的不是嵌套列表
            retDataSet.append(reducedFeatVec)  # 一个列表，切割成了含2个列表的嵌套列表
    return retDataSet

#测试：
# myDat,labels=createDataSet()
# print(myDat)
# print(splitDataSet(myDat,1,1))  # 若第2个特征值=1，则切分且返回

# ---------调用上面的函数，选择最好的数据集划分方式------------------------------------
def chooseBestFeatureToSplit(dataSet):   # 数据集要求是嵌套列表，且所有的元素列表长度相同；最后一列是类别标签
    numFeatures = len(dataSet[0]) -1   # 特征数量
    baseEntropy = calcShannonEnt(dataSet)  # 计算初始数据集的熵，用于对比后续的信息增益
    bestInfoGain = 0.0                   # 初始信息增益为0
    beatFeature = -1                   # 建立特征索引
    for i in range(numFeatures):   # 遍历每个特征
        print('第 %d 个特征' % (i+1))
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)     # 创建唯一的分类标签列表。set方法是最快的
        newEntropy = 0.0
        for value in uniqueVals:  # 遍历当前所选特征向量的所有唯一属性，计算每种划分方式的熵
            subDataSet  = splitDataSet(dataSet,i,value) # 按照第i个特征=value 划分出数据子集
            prob = len(subDataSet)/float(len(dataSet))  # 计算子集频率（概率）
            newEntropy += prob * calcShannonEnt(subDataSet) # 计算新的熵：加和所有子集的熵
        infoGain = baseEntropy - newEntropy  #（整个）数据集的信息增益。信息增益表示信息有序度增加，即熵的减少或者数据无序度的减少
        
        if (infoGain > bestInfoGain):          # 比较信息增益大小，取大者（使信息变得最有序的）
            bestInfoGain = infoGain
            bestFeature = i   # 返回最好特征的索引值
    return bestFeature

# #测试：
#myDat,labels=createDataSet()#
#print(chooseBestFeatureToSplit(myDat))
#print(myDat)
# 结果是索引为0的特征，即第1个特征最好。
# myDat: [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
# 按第1个特征切，即第一个特征值为1的放一组，为0的放另一组：
    #         [[1, 1, 'yes'],
    #          [1, 1, 'yes'],
    #          [1, 0, 'no'],   该组内有1个不一致
    #    ---------------------
    #          [0, 1, 'no'],
    #          [0, 1, 'no']]
#按第2个特征切，即第2个特征为1的放一组，为0的放另一组：
    #        [[1,1, 'yes'],
    #         [1,1, 'yes'],
    #         [0,1, 'no'],
    #         [0,1, 'no'],    # 该组内2个不一致
    #       -----------------
    #         [1,0, 'no']]

# ------------终叶节点类别标签不唯一时，投票表决所属类别------------------------------------------

def majorityCnt(classList):  # 分类名称列表
    classCount={}       # 创建键值唯一的字典，键为分类标签名：值为投票数
    for vote in classList: # 遍历类标签，统计各标签出现的频数
        if vote not in classsCount.keys():  # 若该类标签不在classCount中，添加进去，并计票
            classCount[vote] = 0
            classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    # 排序投票结果。python内建函数sorted(list等可迭代类型,函数，倒序）。python3 中dict.itmes() 迭代 字典的键值对
    return sortedClassCount[0][0]     # 返回出现得票最多的类标签

# ----------- 递归构建决策树 --------------------------------

def createTree(dataSet,labels):    #  传入参数：数据集，标签列表
    classList = [example[-1] for example in dataSet] # 数据集最后一列为标签，返回该列列表
    if classList.count(classList[0]) == len(classList): # list.count(value) 统计列表中某值出现的次数
        return classList[0]  # 递归函数的第一个终止条件：所有样本的标签完全相同，被划为同一类。如classList=['yes','yes','no'],则会进行迭代
    if len(dataSet[0]) == 1: # 第二个终止条件: 遍历完所有特征。如 len([1，1，'yes']) ==3,迭代；len(['yes'])==1,特征用完
        return majorityCnt(classList) # 特征用完仍不能划分成仅含唯一类别的分组，投票

    # 得到最优特征
    bestFeat = chooseBestFeatureToSplit(dataSet) # 最优分类特征的索引
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}     # 每次的最优决策：切割数据集所选的最优属性标签
    del (labels[bestFeat])         # 切割后，排除该标签，在子集中继续切割
    featValues = [example[bestFeat] for example in dataSet]  # 每个样本的最优特征值 列表
    uniqueVals = set(featValues)   # 去重，最优特征值 列表
    # 递归建树：
    for value in uniqueVals:  # 遍历当前选择的最优特征的所有属性值
        subLabels = labels[:] # 子集的所有分类标签. python中，函数参数是list类型，按引用的方式传递。确保每次调用createTree时原始列表不变
        # 在每个子集上递归调用createTree(),返回值插入字典myTree
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree

# 测试：
# myDat,labels=createDataSet()
# myTree = createTree(myDat,labels)
# print(myTree)
    # 返回嵌套字典：键的值若是【类标签】，则该子节点是【叶节点】；若是【字典】，则子节点是【判断节点】

#--------- 3.3 使用决策树进行分类 --------------------------

def classify(inputTree,featLabels,testVec):  # 参数为已经训练好的决策树模型(dict)，标签向量y(list)，测试集向量(list)
    firstStr = list(inputTree.keys())[0]  # 选取决策节点
    print('判断节点：',firstStr)
    secondDict = inputTree[firstStr]     # 决策节点下的值，是dict或str
    print('判断节点的值：',secondDict)
    featIndex = featLabels.index(firstStr) # 将标签字符串转换成索引，用于判断该属性是特征的第一个属性（维度）还是第二个属性
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key],featLabels,testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

# 测试：
# myDat,labels=createDataSet()
# print('labels: ',labels)
# myTree={'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
# print('myTree',myTree)
# classify(myTree,labels,[1,0])
# classify(myTree,labels,[1,1])

# --------训练后的分类器的存储-------------------------------------------------------

# 重复训练、构造决策树会很耗时，因此需要存储已经训练好的分类器，方便即时调用。
# To get around this, you’re going to use a Python module, which is properly named pickle,to serialize objects。
# Serializing objects allows you to store them for later use.
# Serializing can be done with 【any object】, and dictionaries work as well.

def storeTree(inputTree, filename):
	import pickle
	fw = open(filename,'w')
	pickle.dump(inputTree,fw)
	fw.close()
	
def grabTree(filename):
	import pickle
	fr = open(filename)
	return pickle.load(fr)
	
# 测试：
# storeTree(myTree, 'classifierStorage.txt')
# print(grabTree('classifierStorage.txt'))
