# -*- coding: utf-8 -*-

# z = wx  (w是向量）
# sigmoid(z) = 1 / (1+exp(-z))
# 已有数据集x向量，求解最佳w:
# 求最大值 Gradient Ascent:  w := w + α·▽f(w)  α是步长
# 求最小值 Gradient Descent: w := w - α·▽f(w)
# 初始w_0 =1，迭代到某个条件停止

import numpy as np

# --------------- 梯度上升算法 ----------------

# 数据集有100个样本点，每个点是2维向量（X1,X2)

def loadDataSet():   # 函数作用：读取文件并转换为可用于计算的数据集，包括训练集X，标签集y
	dataMat = []; labelMat = []
	fr = open('testSet.txt')
	for line in fr.readlines():
		lineArr = line.strip().split()
		dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
		labelMat.append(int(lineArr[2]))
	return dataMat,labelMat
	
def sigmoid(inX):	
	return 1.0/(1+np.exp(-inX))

def gradAscent(dataMatIn,classlabel):
	dataMatrix = np.mat(dataMatIn)  # 转换成矩阵
	labelMat = np.mat(classlabel).transpose()
	m,n = np.shape(dataMatrix)
	alpha = 0.001	# 步长
	maxCycles = 500	# 迭代次数
	weights = np.ones((n,1)) # 初始权值
	for  k in range(maxCycles):
		h = sigmoid(dataMatrix * weights)
		error = (labelMat - h)  # label 一致则为0，不一致则不为0（1或-1）
		weights = weights + alpha * dataMatrix.transpose() * error # 权重有错才更新？？
	return weights  # 返回矩阵，分别对应系数

# 终端执行：
# import logistic
# dataArr,labelMat=logistic.loadDataSet()
# weights=logistic.gradAscent(dataArr,labelMat)
# print(weights)
	
# -------------画出决策边界-------------------------------
	
def plotBestFit(wei):
	import matplotlib.pyplot as plt
	weights = wei#.getA()   # np.matrix.getA() matrix转换成 array。若是随机梯度算法，无需转换
	dataMat,labelMat=loadDataSet()
	dataArr = np.array(dataMat)
	n=np.shape(dataArr)[0]
	xcord1 = []
	ycord1=[]
	xcord2 = []
	ycord2=[]
	for i in range(n):    # 将两类样本点区分成2个数据集
		if int(labelMat[i]) ==1:
			xcord1.append(dataArr[i,1])
			ycord1.append(dataArr[i,2])
		else:
			xcord2.append(dataArr[i,1])
			ycord2.append(dataArr[i,2])
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')  # 先画出散点图
	ax.scatter(xcord2,ycord2,s=30,c='green')		 # 真实的样本点分布：正样本红点，负样本绿点
	x = np.arange(-3.0,3.0,0.1)      	# 再画出直线，直线上各点对应的(x,y)即是样本点坐标(X1,X2)
	y = (-weights[0]-weights[1]*x)/weights[2]  # 拟合得直线 w0 X0 + w1 X1 + w2 X2 = 0 (X0=1)
	ax.plot(x,y)            
	plt.xlabel('X1')   # x轴为X1
	plt.ylabel('X2')   # y轴为X2
	plt.show()
	
# 继上面执行后，终端执行：logistic.plotBestFit(weights)

# ------------------------ 随机梯度上升 -------------------------------------------
# 上面计算量较大，每次迭代都需要遍历整个数据集。因此改进为随机梯度算法：随机遍历一个或几个点来更新系数W.
# 这是一个online learning algorithm，在新样本进来时分类器增量式更新。对应的，一次处理所有数据被称作batch processing

# Pseudo-code:
# Start with the weights all set to 1
# For each piece of data in the dataset:
# 	▽f(w)              # Calculate the gradient of one piece of data
# 	w = w + α·▽f(w)    # Update the weights vector by α * ▽
# Return the weights vector

def stocGradAscent0(dataMatrix,classLabels): # 训练数据集（X,y)
	m,n = np.shape(dataMatrix)  # 样本数量
	alpha = 0.01                
	weights = np.ones(n)    # 初始值设为1
	for i in range(m):      # 随机遍历【1个点】来更新；之前是遍历整个数据集【向量】
		h = sigmoid(sum(dataMatrix[i] * weights)) # 这里 h,error是数值，类型是array；不是向量
		error = classLabels[i] - h                # 运算也不是矩阵运算
		weights = weights + alpha * error * dataMatrix[i]
	return weights

# 终端：
# from numpy import *
# imp.reload(logistic)   # python3 中reload 放到了imp 内建模块中
# dataArr,labelMat=logistic.loadDataSet()
# weights=logistic.stocGradAscent0(array(dataArr),labelMat)
# logistic.plotBestFit(weights) # 注意先注释该函数中的getA()
# 最佳拟合线比原先稍差，错分了约1/3

# **     直接比较梯度算法和随机梯度算法的结果是不公平的，因为迭代次数不一样。
# **     判断优化算法优劣：看它是否收敛，即蚕食是否达到稳定值，是否还会不断变化。

# ------- 对此，修改随机梯度算法，使运行200次------------------------------------------------

def stocGradAscent1(dataMatrix,classLabels,numIter=150):
	import random
	m,n = np.shape(dataMatrix)
	weights = np.ones(n)
	for j in range(numIter):	# j是迭代次数
		dataIndex = list(range(m))	# 样本编号，即样本在矩阵中的位置
		for i in range(m):      # i是样本下标，不同于样本编号，表示本次(j)迭代中第i个选出来的样本
			alpha = 4/(1.0+j+i) + 0.01  # 步长随迭代次数减小，越接近谷底越小，但不会到0
			randIndex = int(random.uniform(0,len(dataIndex))) # 随机取样更新，减少波动
			h = sigmoid(sum(dataMatrix[randIndex]*weights))
			error = classLabels[randIndex] - h
			weights = weights + alpha * error * dataMatrix[randIndex]
			del(dataIndex[randIndex]) # python3 中range不支持item删除，需要前面list()
	return weights

#终端测试：
# import logistic
# import numpy as np
# dataArr,labelMat=logistic.loadDataSet()
# weights = logistic.stocGradAscent1(np.array(dataArr),labelMat,500)
# logistic.plotBestFit(weights)

#---------案例：预测马疝病的死亡率----------------------------------------------

def classifyVector(inX,weights):  # 0/1分类器：指示函数
	prob = sigmoid(sum(inX * weights)) #（每个特征向量 * 各回归系数）的线性组合
	if prob > 0.5:
		return 1.0
	else:
		return 0
def colicTest():
	frTrain = open('horseColicTraing.txt') 
	frTest = open('horseColicTest.txt')	  
	trainingSet = []                    
	trainingLabels = []
	for line in frTrain.readlines():     # 先处理训练集：逐行读入训练集
		currLine = line.strip().split('\t') # 格式化每行
		lineArr = []
		for i in range(21):      # 有21个特征（列），共22列，最后一列是label
			lineArr.append(float(currLine[i])) # 每一行（列表）的21个特征(item)转换成数值型
		trainingSet.append(lineArr)            # 得到训练集
		trainingLabels.append(float(currLine[21]))  # 训练集最后一列labels 单独拎出来
	
	trainWeights = stocGradAscent1(np.array(trainingSet),trainingLabels,500) # 用随机梯度算法训练得最佳参数
	errorCount = 0  # 统计犯错数
	numTestVec = 0.0  # 统计测试集容量
	for line in frTest.readlines():  # 处理测试集，和前面相同
		numTestVec += 1.0
		currLine = line.strip().split('\t')
		lineArr = []
		for i in range(21):
			lineArr.append(float(currLine[i]))
		if int(classifyVector(np.array(lineArr),trainWeights)) != int(currLine[21]):
			errorCount +=1    # 用训练集训练后得到参数trainWeights，代入测试集，比较测试结果和真实结果，统计犯错数
			
	errorRate = (float(errorCount)/numTestVec)  # 错误率=犯错数/测试集容量
	print('the error rate of this test is : %f' % errorRate)
	return errorRate   # 返回错误率
	
def multiTest():   # 实现多轮测试：调用colicTest()10次，取平均错误率
	numTests = 10
	errorSum = 0.0
	for k in range(numTests):
		errorSum += colicTest()
	print('after %d iterations the average error rate is: %f' % (numTests,errorSum/float(numTests)))

# 终端执行：logistic.multiTest()



