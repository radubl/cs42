import os
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.ensemble import VotingClassifier

from sklearn.cross_validation import cross_val_score
import numpy as np
import pandas as pd

with open('features/target.npy', 'rb') as f:
	target = np.load(f)

def showImage(original, title):
	ax1 = plt.subplot(1,1,1)
	ax1.imshow(original, cmap=cm.binary)
	ax1.set_title(title)
	plt.show()


models = {
# 'random_forest_500' : RandomForestClassifier(n_estimators=500,n_jobs = -1),
# 'random_forest_500d5' : RandomForestClassifier(max_depth=5, n_estimators=500,n_jobs = -1),
# 'random_forest_500feat' : RandomForestClassifier(max_features = "log2", n_estimators=500,n_jobs = -1),
# 'random_forest_500leaf' : RandomForestClassifier(min_samples_leaf = 5, n_estimators=500,n_jobs = -1),
# 'random_forest_1000' : RandomForestClassifier(n_estimators=1000,n_jobs = -1),
# 'random_forest_1500' : RandomForestClassifier(n_estimators=1500,n_jobs = -1),
# 'random_forest_2500' : RandomForestClassifier(n_estimators=2500,n_jobs = -1),
# 'random_forest_5000' : RandomForestClassifier(n_estimators=5000,n_jobs = -1)
# 'kNN10' : KNeighborsClassifier(n_neighbors=10,n_jobs = -1),
# 'kNN5' : KNeighborsClassifier(n_neighbors=5,n_jobs = -1)
# 'hardVoting' : VotingClassifier(estimators=[('lr', RandomForestClassifier(n_estimators=5000,n_jobs = -1)), ('rf', svm.LinearSVC(C=0.1)), ('gnb', KNeighborsClassifier(n_neighbors=5,n_jobs = -1))], voting='hard')
# 'svc_rbf_C=1' :  svm.SVC(kernel='rbf',C=1.0),
# 'svc_rbf_C=0.1' :  svm.SVC(kernel='rbf',C=0.1),
'svc_poly_C=1.0' :  svm.SVC(kernel='poly',C=1.0),
# 'linSVC_C0.1': svm.LinearSVC(C=0.1),
# 'linSVC_C1': svm.LinearSVC(),
# 'adaB' : AdaBoostClassifier()
}

cut = int(len(target) / 10)

def makeSubmission(model,dataset):

	print "Generating submission"

	with open('features/'+ dataset + '.npy', 'rb') as f:
		features = np.load(f)

	train = np.asarray(features[0])
	test = np.asarray(features[1])

	print "Feature_set " + dataset

	shuffle(train,target)

	model.fit(train,target)

	pred = model.predict(test)

	np.savetxt('new_sub_KNN_0.csv', np.c_[range(1,len(test)+1),pred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')

# makeSubmission(KNeighborsClassifier(n_neighbors=5,n_jobs = -1), 'gaussianed_0.8_deskewed')

# exit()

def classifyFastAllFeatures(featureFiles):

	output = ''

	for key, value in models.iteritems():

		print "\nModel " + key

		for featureFile in featureFiles:

			with open('features/'+ featureFile + '.npy', 'rb') as f:
				features = np.load(f)

			train = np.asarray(features[0])

			print "Feature_set " + featureFile

			shuffle(train,target)

			model = value

			model.fit(train[:cut*2],target[:cut*2])

			print model.score(train[-cut:],target[-cut:])

			# all dataset

			# model.fit(train[:cut*9],target[:cut*9])

			# print model.score(train[-cut:],target[-cut:])
	



# classifyFastAllFeatures(['gaussianed_0.2_deskewed', 'gaussianed_0.5_deskewed', 'gaussianed_0.8_deskewed'])

# classifyFastAllFeatures(['cannied_1_deskewed', 'cannied_3_deskewed', 'cannied_3_deskewed'])


# classifyFastAllFeatures(['vanilla'])

# classifyFastAllFeatures(['deskewed'])

# classifyFastAllFeatures(['gaussianed_0.2_deskewed'])

# classifyFastAllFeatures(['gaussianed_0.8_deskewed'])

classifyFastAllFeatures(['hog_deskewed'])

# classifyFastAllFeatures(['gaussianed_2_deskewed'])

# with open('features/gaussianed_2_deskewed.npy', 'rb') as f:
# 	features = np.load(f)

# matrix = features[0][42].reshape(28,28)
# showImage(matrix, 'Gaussian')

exit()


with open('features/hog_gaussianed_deskewed.npy', 'rb') as f:
		features = np.load(f)

train = np.asarray(features[0])
test = np.asarray(features[1])

for key, value in models.iteritems():

	print "Model " + key

	model = value

	scores = cross_val_score(value, train, target, cv=3)
	print scores

	# model.fit(train[:cut*9],target[:cut*9])

	# print model.score(train[-cut:],target[-cut:])
	
	model.fit(train,target)

	pred = model.predict(test)

	np.savetxt('submission_linearSVG_hog_gaussianed_deskewed.csv', np.c_[range(1,len(test)+1),pred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')
	



	# print scores

# rf.fit(train, target)
# pred = rf.predict(test)
