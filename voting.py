import pandas as pd
import numpy as np
import itertools
import operator
import csv

def getListFromCsv(csvName):
	reader = csv.DictReader(open(csvName))

	result = {}
	for row in reader:
	    for column, value in row.iteritems():
	        result.setdefault(column, []).append(value)
	return result

def most_common(L):
  # get an iterable of (item, iterable) pairs
  SL = sorted((x, i) for i, x in enumerate(L))
  # print 'SL:', SL
  groups = itertools.groupby(SL, key=operator.itemgetter(0))
  # auxiliary function to get "quality" for an item
  def _auxfun(g):
    item, iterable = g
    count = 0
    min_index = len(L)
    for _, where in iterable:
      count += 1
      min_index = min(min_index, where)
    # print 'item %r, count %r, minind %r' % (item, count, min_index)
    return count, -min_index
  # pick the highest-count/earliest item
  return max(groups, key=_auxfun)[0]


d1 = getListFromCsv("new_sub_KNN_0.csv")
d3 = getListFromCsv("submission_dbnn.csv")
d4 = getListFromCsv("submission_linearSVG_hog_gaussianed_deskewed.csv")


votes = []

for x in xrange(0,len(d1['Label'])):
	secVotes = []

	secVotes.append(int(most_common([d1['Label'][x],d3['Label'][x],d4['Label'][x]])))
	secVotes.append(int(most_common([d4['Label'][x],d1['Label'][x],d3['Label'][x]])))
	secVotes.append(int(most_common([d3['Label'][x],d4['Label'][x],d1['Label'][x]])))

	votes.append(most_common(secVotes))

np.savetxt('voting3.csv', np.c_[range(1,len(votes)+1),votes], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')
