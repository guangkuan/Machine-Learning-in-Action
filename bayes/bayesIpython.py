import bayes

#for the loadDataSet
listOposts, listClasses = bayes.loadDataSet()
myVocabList = bayes.createVocabList(listOposts)
myVocabList
bayes.setOfWords2Vec(myVocabList, listOposts[0])
bayes.setOfWords2Vec(myVocabList, listOposts[3])

#for the trainNB0
from imp import reload
reload(bayes)
listOposts, listClasses = bayes.loadDataSet()
myVocabList = bayes.createVocabList(listOposts)
trainMat = []
for postinDoc in listOposts:
	trainMat.append(bayes.setOfWords2Vec(myVocabList, postinDoc))
p0V, p1V, pAb = bayes.trainNB0(trainMat, listClasses)
pAb
p0V
p1V

#for the classifyNB
reload(bayes)
bayes.testingNB()

#for the bagOfWords2VecMN
mySent = 'This book is the best book on Python or M.L. I have ever laid eyes upon.'
mySent.split()

import re
regEx = re.compile('\\W*')
listOfTokens = regEx.split(mySent)
listOfTokens

[tok for tok in listOfTokens if len(tok) > 0]

[tok.lower() for tok in listOfTokens if len(tok) > 0]

emailText = open('email/ham/6.txt').read()
listOfTokens = regEx.split(emailText)

#for the spamTest
bayes.spamTest()

#for the feedparser
import feedparser
ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
ny['entries']
len(ny['entries'])
history

#for the calcMostFreq
reload(bayes)
ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
vocabList, pSF, pNY = bayes.localWords(ny, sf)

#for the getTopWords
reload(bayes)
bayes.getTopWords(ny, sf)