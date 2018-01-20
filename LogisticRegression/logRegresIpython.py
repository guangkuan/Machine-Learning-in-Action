#for the loadDataSet
import logRegres
dataArr, labelMat = logRegres.loadDataSet()
wei = logRegres.gradAscent(dataArr, labelMat)

#for the gradAscent
from imp import reload
reload(logRegres)
weights1 = wei.getA()
logRegres.plotBestFit(weights1)

#for the stocGradAscent0
from numpy import *
reload(logRegres)
dataArr, labelMat = logRegres.loadDataSet()
weights2 = logRegres.stocGradAscent0(array(dataArr), labelMat)
logRegres.plotBestFit(weights2)

#for the stocGradAscent1
reload(logRegres)
dataArr, labelMat = logRegres.loadDataSet()
weights3 = logRegres.stocGradAscent1(array(dataArr), labelMat)
logRegres.plotBestFit(weights3)

#for the classifyVector
reload(logRegres)
logRegres.multiTest()
