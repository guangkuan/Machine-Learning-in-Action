# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 22:24:23 2017

@author: Administrator
"""

import kNN
group,labels = kNN.createDataSet()
group
labels
kNN.classify0([0, 0], group, labels, 3)

#for the file2matrix
from imp import reload
reload(kNN)
datingDataMat,datingLabels = kNN.file2matrix('datingTestSet.txt')
datingDataMat
datingLabels[0:20]

#for the picture
import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,1],datingDataMat[:,2])
plt.show()
ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*array(datingLabels),15.0*array(datingLabels))
ax.scatter(datingDataMat[:,0],datingDataMat[:,1],15.0*array(datingLabels),15.0*array(datingLabels))

#for the autoNorm
reload(kNN)
norMat, ranges, minVals = kNN.autoNorm(datingDataMat)
norMat
ranges
minVals

#for the datingClassTest
kNN.datingClassTest()

#for the clasdifyPerson
kNN.classifyPerson()

#for the img2vector
testVector = kNN.img2vector('testDigits/0_13.txt')
testVector[0,0:31]
testVector[0,31:63]

#for the handwritingClassTest
kNN.handwritingClassTest()