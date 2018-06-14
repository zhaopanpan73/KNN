# 书读百遍其义自见。每次思考KNN算法的特性都有不同的领悟，下一次我要讲给小伙伴听
# 下面是根据自己的理解写的算法

# 建立一个KNN模型应该要的准备:
# (1) 收集数据--->这一步都是找的现成的
# (2) 准备数据:  使用Python解析文本文件。主要是从文件中分割出数据集和标签
# (3) 分析数据:  使用Matplotlib画出二维扩散图
# (4) 训练算法:  KNN算法不需要这一步
# (5) 测试算法:  测试集:从数据中分离的带有标签的数据。如果预测与实际的标签相同，则分类正确。若不同则分类错误
# (6) 使用算法:  输入测试的特征数据，计算距离，然后进行类别判断。


import numpy as np
from numpy import array
# (1) 收集数据

# (2) 准备数据:使用python解析文本文件---->生成数据属性和标签
filename="data.txt"
# 加载数据
def  loadData(filename):
     with open(filename) as f:
         arrayOfLines=f.readlines()
         nSamples=len(arrayOfLines)  # f.readlines()按行读取全部的文本  获取样本的个数
         nAttr=3  # 获取属性的个数
         data=np.zeros((nSamples,nAttr))
         label=[]
         i=0
         for line in arrayOfLines:
             line=line.strip()        # 去掉所有的回车符
             ListFromline=line.split('\t')    # 用tab (\t)分割得到的数据的列表
             data[i,:]=ListFromline[:-1]
             label.append(int(ListFromline[-1]))  #   ListFromline[-1]得到的是'3' ---->是个字符   ListFromline[-1:] 得到的是['3'] ----->这是个list
             i+=1

     return  data ,label


# 归一化
def autoNorm(dataSet):
    minValues=dataSet.min(0)   # 0 表示求每一列的最小值
    maxValues=dataSet.max(0)   # 0 表示求每一列的最小值
    ranges=maxValues-minValues
    normData=np.zeros(dataSet.shape)
    nSamples,nAttr=dataSet.shape
    # 将数据扩展到与数据集同一个形状，便于计算
    normData=dataSet-np.tile(minValues,(nSamples,1))  # (nSamples,1)在行上重复nSamples次，在列上重复1次
    normData = normData / np.tile(maxValues, (nSamples, 1))   # /逐个元素相除，相当于np.divide
    return  normData,ranges,minValues


# (3) 分析数据:  使用Matplotlib画出二维扩散图
def  showScatterOfData():
    data,label=loadData(filename)
    import matplotlib
    import matplotlib.pyplot as plt
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(data[:,0],data[:,1],15.0*np.array(label),15.0*np.array(label))  # 根据数据的label来给点上色
    plt.show()

# 看看示意图的效果
showScatterOfData()

# (4) 训练算法:  KNN算法不需要这一步


# (5) 测试算法:  测试集:从数据中分离的带有标签的数据。如果预测与实际的标签相同，则分类正确。若不同则分类错误
def classifyAlgorithm(testX,dataSet,labels,k):
     import operator
     # 下面就开始求距离啦
     nSamples,Attr=dataSet.shape
     Sub=np.tile(testX,(nSamples,1))-dataSet  # 逐个元素相减
     Square=Sub**2
     sqDistance=Square.sum(axis=1)
     distance=sqDistance**0.5  # 这样算是求平方
     sortedDistanceIndex=distance.argsort()
     classCount={}
     for i in range(k):
         voteLabel=labels[sortedDistanceIndex[i]]
         classCount[voteLabel]=classCount.get(voteLabel,0)+1  # 这个语法很好用
     sortClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
     return sortClassCount[0][0]

def ClassTest():
    testRatio=0.6
    testData,testLabel= loadData("data.txt")
    normData,ranges,minValues=autoNorm(testData)
    nSamples,Attr=normData.shape
    numTest=int(testRatio*nSamples)
    errorCount=0.0
    for i in range(numTest):
        result=classifyAlgorithm(normData[i,:],normData[numTest:nSamples,:],testLabel[numTest:nSamples],3)
        print("predict label :%d   real label : %d  "%(result,testLabel[i]))
        if result!=testLabel[i]:
            errorCount+=1
    print("The total error is %d " % (errorCount/numTest))

ClassTest()