
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from skimage.feature import canny
from skimage.filters import gaussian
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm

def getCannied(name,reshapeSize):
	with open('features/'+name, 'rb') as f:
		features = np.load(f)

	for sigma in xrange(1,4):
		print 'Canniyng '+name

		print 'Train data'

		train = []
		for feature in features[0]:
			matrix = feature.reshape((reshapeSize, reshapeSize)).astype(np.uint8)
			train.append(canny(matrix, sigma=sigma).ravel())

		print 'Test data'
		test = []
		# for feature in features[1]:
		# 	matrix = feature.reshape((reshapeSize, reshapeSize)).astype(np.uint8)
		# 	test.append(canny(matrix, sigma=2).ravel())

		print 'Saving features'
		data = np.array([train, test])

		with open('features/cannied_' + str(sigma)+ '_' + name  , 'wb') as f:
			np.save(f, data)


def getGaussian(name,reshapeSize):
	with open('features/'+name, 'rb') as f:
		features = np.load(f)

	for sgm in [2]:
		print 'Gaussianing with ' + str(sgm)+ '_' +name

		print 'Train data'

		train = []
		for feature in features[0]:
			matrix = feature.reshape((reshapeSize, reshapeSize)).astype(np.uint8)
			train.append(gaussian(matrix,sigma=sgm).ravel())
		print 'Test data'
		test = []
		for feature in features[1]:
			matrix = feature.reshape((reshapeSize, reshapeSize)).astype(np.uint8)
			test.append(gaussian(matrix,sigma=sgm).ravel())

		print 'Saving features'
		data = np.array([train, test])

		with open('features/gaussianed_' + str(sgm)+ '_' + name  , 'wb') as f:
			np.save(f, data)

# getGaussian('vanilla.npy',28)
getGaussian('deskewed.npy',28)

# getCannied('vanilla.npy',28)
# getCannied('deskewed.npy',28)
