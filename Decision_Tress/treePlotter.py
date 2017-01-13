import matplotlib.pyplot as plt

# ------------ 如何绘制节点及注释 ----------------------------
# 定义文本框和箭头格式。参数以字典的形式传入。
decisionNode = dict(boxstyle='sawtooth',fc='0.8')  # 决策节点为 锯齿框
leafNode = dict(boxstyle='round4',fc='0.8')        # 叶节点为 圆角框
arrow_args = dict(arrowstyle='<-')                 # 箭头格式


# def createPlot():  # 创建作图的区域，包括区域尺寸、各节点坐标位置之类的布局(后文会继续补全修正该函数）
#     fig = plt.figure(1,facecolor='white')  # create a new figure，底色为白色
#     fig.clf()                              # clear the figure
#     createPlot.ax1=plt.subplot(111,frameon=False) # 在figure（绘图区）中创建子绘图区，参数表示子图尺寸。注意.ax1的赋值形式
                                                    # Python默认所有变量全局有效
#     plotNode('a decision node',(0.5,0.1),(0.1,0.5),decisionNode) # 画出决策节点
#     plotNode('a leaf node',(0.8,0.1),(0.3,0.8),leafNode)         # 画出叶节点
#     plt.show()

# 在绘图区中，执行实际的绘图动作，主要是绘制各类节点、箭头并注释
def plotNode(nodeTxt,centerPt,parentPt,nodeType): # 参数为节点文本（字符串），箭头头部坐标，箭头尾部坐标，节点类型(决策或叶节点）
    createPlot.ax1.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',xytext=centerPt, #该函数需要一个绘图区，在其他函数中定义
                            textcoords='axes fraction',va='center',ha='center',            #即基于createPlot.ax1进行
                            bbox=nodeType,arrowprops=arrow_args)                     # .annotate注释。参数见文档

#测试：
# createPlot()

# -------------- 绘制整棵树 ----------------------------------------------
# 如何放置所有的树节点：我们必须知道有多少叶节点，以便确定 x轴的长度
#                     还需要知道树有多少层，以便确定 y轴的高度

# 遍历整颗树，递归获取叶节点数目：
# 已知树信息：{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}

def getNumLeaves(myTree):
    numLeaves = 0
    firstStr = list(myTree.keys())[0] # 取字典的第一个键。python3 中dict.keys() 不再返回列表。
    secondDict = myTree[firstStr] # 第一个key的value
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':    # 测试节点的数据类型是否为字典
            numLeaves += getNumLeaves(secondDict[key]) # 是字典，则该节点是判断节点，需要递归调用getNumLeaves
        else:
            numLeaves += 1  # 不是字典，则是叶子节点。累计。
    return numLeaves

# 获取树的层数（深度）：统计判断节点数量，再加上最后全是叶节点的一层
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth >maxDepth:
            maxDepth = thisDepth
    return maxDepth

# 测试：
#myTree={'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
# print(getNumLeaves(myTree))
# print(getTreeDepth(myTree))

# ----------结合以上，绘制完整的树 ---------------------
def plotMidText(cntrPt,parentPt,txtString): # 在父子节点间填充文本信息0/1
    xMid = (parentPt[0]-cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid,yMid,txtString) # .text： Add text to the axes.

def plotTree(myTree,parentPt,nodeTxt):   # 执行具体的绘图动作，参数为树信息、设定父节点（第一个决策特征）坐标、节点文本
    numLeaves = getNumLeaves(myTree)     # 计算树的宽度和高度，以定位节点位置
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeaves))/2.0 / plotTree.totalW,plotTree.yOff)
    plotMidText(cntrPt,parentPt,nodeTxt)
    plotNode(firstStr,cntrPt,parentPt,decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        print(type(secondDict[key]).__name__)
        if type(secondDict[key]).__name__=='dict':
            plotTree(secondDict[key],cntrPt,str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key],(plotTree.xOff,plotTree.yOff),cntrPt,leafNode)
            plotMidText((plotTree.xOff,plotTree.yOff),cntrPt,str(key))
    plotTree.yOff = plotTree.yOff + 1.0/ plotTree.totalD

def createPlot(inTree):  # 主函数，调用plotTree,plotTree又调用前面的函数
    fig = plt.figure(1,facecolor='white')
    fig.clf()
    axprops = dict(xticks=[],yticks=[])
    createPlot.ax1=plt.subplot(111,frameon=False,**axprops)
    plotTree.totalW = float(getNumLeaves(inTree))  # 全局变量，存储树宽
    plotTree.totalD = float(getTreeDepth(inTree))  # 全局变量，存储树高
    plotTree.xOff = -0.5 / plotTree.totalW         # 追踪已绘制节点的位置，以及放置下一节点的位置
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), ' ')     # 父节点坐标（0.5, 1.0)
    plt.show()

# 测试：
# myTree={'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
# createPlot(myTree)