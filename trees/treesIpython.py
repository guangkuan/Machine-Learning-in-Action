# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 22:01:54 2017

@author: Administrator
"""
#for the calcShannonEnt
import trees

#for the createDataSet()
from imp import reload
reload(trees)
myDat, labels = trees.createDataSet()
myDat
trees.calcShannonEnt(myDat)
myDat[0][-1] = 'maybe'
myDat
trees.calcShannonEnt(myDat)

#for the splitDataSet
a = [1, 2, 3]
b = [4, 5, 6]
a.append(b)
a
a = [1, 2, 3]
a.extend(b)
a
reload(trees)
myDat, labels = trees.createDataSet()
myDat
trees.splitDataSet(myDat, 0, 1)
trees.splitDataSet(myDat, 0, 0)

#for the chooseBestFeatureToSplit
reload(trees)
myDat, labels = trees.createDataSet()
trees.chooseBestFeatureToSplit(myDat)
myDat

#for the createTree
reload(trees)
myDat, labels = trees.createDataSet()
myTree = trees.createTree(myDat, labels)
myTree

#for the classify
myDat, labels = trees.createDataSet()
labels
myTree = treePlotter.retrieveTree (0)
myTree
trees.classify(myTree, labels, [1, 0])
trees.classify(myTree, labels, [1, 1])

#for the storTree
trees.storeTree(myTree, 'classifierStorage.txt')
trees.grabTree('classifierStorage.txt')