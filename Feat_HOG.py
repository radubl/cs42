# Import the modules
from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
import os
import numpy as np
import pandas as pd

def getHOG(name,reshapeSize):
	with open('features/'+name, 'rb') as f:
		features = np.load(f)

	features
	train = []

	print 'Hogging '+name

	print 'Train data'

	for feature in features[0]:
	    fd = hog(feature.reshape((reshapeSize, reshapeSize)), orientations=20, pixels_per_cell=(2, 2), cells_per_block=(1, 1), visualise=False)
	    train.append(fd)

	train = np.array(train, 'float64')

	print 'Test data'
	test = []
	for feature in features[1]:
	    fd = hog(feature.reshape((reshapeSize, reshapeSize)), orientations=20, pixels_per_cell=(2, 2), cells_per_block=(1, 1), visualise=False)
	    test.append(fd)

	test = np.array(test, 'float64')

	print 'Saving features'
	data = np.array([train, test])

	with open('features/hog_' + name , 'wb') as f:
		np.save(f, data)


with open('features/target.npy', 'rb') as f:
	features = np.load(f)


# classes = np.zeros(10)

# for cls in features:
# 	classes[cls] += 1

# summ = 0
# for x in classes:
# 	summ +=x
# 	print x/420

# print summ

# exit()


getHOG('deskewed.npy',28)