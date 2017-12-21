# for the createPlot
import treePlotter
treePlotter.createPlot()

# for the getNumLeafs
from imp import reload
reload(treePlotter)
treePlotter.retrieveTree(1)
myTree = treePlotter.retrieveTree(0)
treePlotter.getNumLeafs(myTree)
treePlotter.getTreeDepth(myTree)

# for the plotMidText
reload(treePlotter)
myTree = treePlotter.retrieveTree(0)
treePlotter.createPlot(myTree)
myTree['no surfacing'][3] = 'maybe'
myTree
treePlotter.createPlot(myTree)