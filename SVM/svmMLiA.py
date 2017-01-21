# 理论补充：周志华《机器学习》+ 台大《机器学习技法》


# 基本概念------------
# separating hyperplane 分隔超平面：2维以上的数据集，分隔数据的就不再是一条线、一个平面。如1024维的数据集，要用1023维分割。即决策边界。
# support vector支持向量（SV）：离分隔超平面最近的那些点。
# margin间隔： SVM分类器，找到某个决策边界，希望数据点离决策边界越远越好。找到SV，确保它们离分隔面的距离尽可能远，这个距离即间隔margin。

# 要做的：寻找最大间隔-----------
# 如何求解数据集的最佳分隔直线：
# 分隔超平面的公式： wTx+b			  **
# 点到平面的距离：|wTx+b|/ ||w||     **

# 求解最佳化问题(求解W,b)----------
# 输入数据--> 分类器(f(wTx+b) 类似Sigmoid函数的作用) --> 输出类别label（+1或-1）
# 与Logistic不同，logistic输出标签为+1或0
# 这里用-1和+1是因为：只差一个符号，方便数学上处理，可以用一个统一公式来表示间隔（距离），距离无所谓分类是正是负。
	# 当计算间隔距离并确定分隔面位置时，间隔通过 label*(wTx+b)计算得到。   	**
	# 如果数据点为+1类，且离分隔面很远,那么(wTx+b)是个很大的正数，同时label*(wTx+b)也是个很大的正数；
	# 若为-1类，且间隔很大，则label*(wTx+b)仍是个很大的正数。

# 现在的目标是找到 定义分类器的W、b。为此，我们必须找到离超平面最近的数据点——SV，然后将这个距离（间隔）最大化：
	# arg max { min( y*(wTx+b) ) *  (1/||w||) }			**
	#     w,b    n

# 直接求解上述问题很难，所以要转换成容易求解的形式。
# 先观察大括号内的部分。对乘积进行优化是件很讨厌的事，因此我们要做的是【固定其中一个因子而最大化其他因子】。
# 如果令所有SV的 label*(wTx+b)=1，那么可以求 1/||w|| 的最大值得到最终解
# 但是，并非所有点的label*(wTx+b)=1，只有SV是。而离超平面越远的数据点，label*(wTx+b)越大。

# 该问题是一个带约束条件的优化问题。对此，采用拉格朗日乘子法。
# 约束条件为 label*(wTx+b) ≥ 1.0
# 由于约束条件都是基于数据点的，因此我们就可以将超平面写成数据点的形式。因此，优化目标函数(The optimization function)：
	#  max[ ∑ α - 1/2·∑ y_i·y_j·a_i·a_j < x_i·x_j > ] < >表示向量内积    ** α乘子对应一个数据点
	# s.t. : α ≥ 0，∑α_i·y_i = 0

# 这里有个前提假设：数据100%线性可分。但事实上，数据会存在噪音。因此我们引入slack variable 松弛变量，允许
# 一些点处于分隔面错误的一侧。这样我们的优化目标函数仍然不变，只有约束条件不同：
#  C ≥ α ≥ 0 ，∑α_i·y_i = 0													**
# C用于控制”使间隔最大“和”保证大部分点的函数间隔≥1“这2个目标的权重。

# 在优化算法的实现代码中，常数C是一个参数，我们可以通过调参得到不同的结果。一旦求出了所有α，
# 那么分隔超平面就可以通过这些α 来表达。SVM中的主要工作就是求出这些α。

# SMO algorithm-------------------------------------------------------------
# 有了上述的【优化目标函数】，接下来就是求解得到最优α。只要得到最优α，就能得到w,b,即得最佳分隔超平面。
# 以往人们使用quadratic solvers来求解这个最优化问题。这通常需要花费大量的计算，实现起来非常复杂。
# 在此，我们介绍另一种高效的求解方式——SMO算法：
# SMO算法是将大优化问题 分解为 多个小优化问题 来求解。这些小优化问题往往是很容易求解的，
# 并且对他们进行顺序求解的结果与将它们作为整体来求解的结果是完全一致的。在结果完全相同的同时，SMO算法更快。

# SMO算法的目标 是求出一系列α 和 b ，一旦求出了这些α，就很容易计算出w 并得到分隔超平面

# SMO算法的工作原理是：
# https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-98-14.pdf

# 每次循环中选择2个α 进行优化处理。一旦找到一对合适的α，那么就增大其中一个，同时减小另一个。这里所谓的‘合适’
# 就是指两个α 必须符合一定的条件：
	# 一：两个α 必须在间隔边界之外   
	# 二：2个α 还没有进行过区间化处理或者不在边界上

# ----应用简化版SMO算法处理小规模数据集------------------------先通过简化版理解算法工作思路---
# SMO算法中的外循环determine要优化的最佳α对。简化版会跳过这步，
# 首先在数据集上遍历每个α（每个数据点对应一个乘子α），
# 然后在剩下的α集合中随机选择另一个α，从而构成α对。
# 这里很重要的一点，就是我们要同时改变两个α。因为我们有一个约束条件：∑ α· y = 0. 由于改变一个α 可能导致该
# 约束条件失效，因此我们总是同时改变两个α。
# 为此，我们将构建一个辅助函数，用于在某个区间范围内随机选择一个整数。同时，也需要另一个辅助函数，用于在数值
# 太大时对其进行调整：

# ----SMO算法中的辅助函数-------------------

import numpy as np
import random

def loadDataSet(filename): # 从文档中载入、解析数据集，并格式化  # 数据集为3列：2列特征1列标签
	dataMat = []
	labelMat = []
	fr = open(filename)
	for line in fr.readlines():
		lineArr = line.strip().split('\t')
		dataMat.append([float(lineArr[0]),float(lineArr[1])]) 
		labelMat.append(float(lineArr[2]))					  
	return dataMat,labelMat					# 返回数据集X，和标签集y

dataArr,labelArr = loadDataSet('testSet.txt')

# 辅助函数一：选择α对	
def selectJrand(i,m):  # i 是第一个α的下标 α_i，m是所有α的数目
	j = i        
	while (j==i):      #选择了一个α_i后，在剩下的α集合中随机选择另一个α_j，从而构成α对。
		j = int(random.uniform(0,m))
	return j

# 辅助函数二：调整大于H或小于L的 α值
def clipAlpha(aj,H,L):	
	if aj > H:
		aj = H
	if aj < L:
		aj = L
	return aj
	
#------------------The simplified SMO algorithm-------
# Pseudocode:
# create an alphas vector filled with 0s      				 创建一个α向量并将其初始化为0向量
# while the number of iterations is less than MaxIterations: While 迭代次数小于最大迭代此时时（外循环）
	# for every data vector in the dataset:					 	for 遍历数据集中的每个数据向量（内循环）：
	#	  if the data vector can be optimized:						if 该数据向量可以被优化：	
	#		  select another data vector at random						随机选出另一个数据向量
	#		  optimize the two vectors together							同时优化这2个向量
	#		  if the vectors can't be optimized							if 两个向量都不能被优化:
	#			  break														退出for 循环（内循环）
	# if no vectors were optimized								if 所有向量都没被优化：
	#    increment the iteration count								增加迭代次数，继续下一次循环

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):  #参数：数据集、标签集、常数C、容错率、最大迭代次数
    dataMatrix = np.mat(dataMatIn) 		# 数组转换为矩阵，方便很多数学运算操作
    labelMat = np.mat(classLabels).transpose()  # 类标签转置，得到的是列向量而不是(行)列表。于是每行元素都与数据矩阵中的行一一对应。
    b = 0
    m,n = np.shape(dataMatrix) 			# m行 x n列
    alphas = np.mat(np.zeros((m,1)))    # m个乘子α的列矩阵，初始化为 mx1的0向量
    iter = 0							# iter变量，用于存储 没有任何 α 改变的情况下遍历数据集的次数。该变量达到输入值maxIter时，结束运行。
    while (iter < maxIter):
        alphaPairsChanged = 0			# 用于记录α是否已经进行优化。每次循环中，先设为0，再遍历整个集合
        for i in range(m):				# 遍历整个数据集的α_i（第一个α），即循环m次
            fXi = float(				# 超平面模型：f(x_i) = wx_i + b = α·y·x_i + b，计算后预测分类
            	        np.multiply(alphas,labelMat).T   #np.multiply 元素级乘法
            	      * (dataMatrix*dataMatrix[i,:].T)
            	      + b)
            Ei = fXi - float(labelMat[i])  # 误差 = 预测结果 - 真实结果
            # 如果误差很大，就对该数据点对应的 α 进行优化。
            # 其中，正负间隔都测试，也检测α_i值确保 0 < α_i < C。
            # 由于后面α<0 或>C时将被调整为0或C，所以若等于0或C，它们就在边界上，因而不再能减小或增大，也就不值得再优化
            # 如果误差 Ei很大，则该 α_i值得优化
            if ((labelMat[i]*Ei < -toler) & (alphas[i] < C))| ((labelMat[i]*Ei > toler) & (alphas[i] > 0)):
                j = selectJrand(i,m)      						 # 选出 α_j 与 α_i 配对，之后同时优化
                fXj = float(np.multiply(alphas,labelMat).T    	 # 计算 α_j 对应的预测值（标签）
                          *(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])    				 # 计算α_j 对应的误差Ej
                alphaIold = alphas[i].copy()                     # 浅拷贝？ 将旧值放到新的内存空间中
                alphaJold = alphas[j].copy()
                # 计算L和H，使 0 < α_j < C：   
                if (labelMat[i] != labelMat[j]):        		 # 如果y_i,y_j 一正一负
                    L = max(0, alphas[j]-alphas[i] )             # 下限
                    H = min(C, C+alphas[j]+alphas[i])			 # 上限
                else:											 # 如果y_i,y_j 同侧
                    L = max(0, alphas[j]+alphas[i]-C)
                    H = min(C, alphas[j]+alphas[i])
                if L==H:	
                    print("L==H")
                    continue
                #  # eta 是α_j的最优修改量
                eta = 2.0*dataMatrix[i,:]*dataMatrix[j,:].T  \
                	- dataMatrix[i,:]*dataMatrix[i,:].T 	\
                 	- dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0:   									
                    print("eta>=0")
                    continue
                alphas[j] -= labelMat[j]*(Ei-Ej)/eta           	
                alphas[j] = clipAlpha(alphas[j],H,L)
                if (abs(alphas[j]-alphaJold) < 0.00001):
                    print('j not moving enough')
                    continue
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold-alphas[j])
                b1 = b-Ei-labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T -\
                     labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ei - labelMat[i] *(alphas[i]-alphaIold) \
                    * dataMatrix[i,:] * dataMatrix[j,:].T \
                    - labelMat[j] * (alphas[j] - alphaJold) \
                    * dataMatrix[j,:] * dataMatrix[j,:].T
                if (0 < alphas[i]) & (C > alphas[i]):
                    b = b1
                elif (0 <alphas[j]) & (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                    alphaPairsChanged += 1
                    print('iter: %d i: %d, pairs changed %d' % (iter, i ,alphaPairsChanged))
            if (alphaPairsChanged == 0): # 终止条件：在for 循环之外，检查α是否做了更新
                iter += 1
            else:		# 有更新，iter设为0后继续运行。只有变了MaxIter次且任何α不再改变，退出While
                iter = 0
            print('iteration number: %d' % iter)
        return b,alphas

# 测试： 由于SMO算法的随机性，每次运行结果可能不同
# b,alphas = smoSimple(dataArr,labelArr,0.6,0.001,40)
# print('b:',b)
# print("α: ",alphas[alphas > 0]) # 可以直接观察α矩阵，但是其中0元素太多，在此过滤操作
# print('支持向量的个数：',np.shape(alphas[alphas > 0]))
# for i in range(100):   #了解哪些数据点事支持向量
#     if alphas[i] > 0.0:
#         print('支持向量为：',dataArr[i],labelArr[i])