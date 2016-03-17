# import the necessary packages
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn import datasets
from nolearn.dbn import DBN
import numpy as np

with open('features/target.npy', 'rb') as f:
	target = np.load(f)

with open('features/hog_gaussianed_0.8_deskewed.npy', 'rb') as f:
	feat = np.load(f)

train = feat[0]
test = feat[1]

# scale the data to the range [0, 1] and then construct the training
# and testing splits
# (trainX, testX, trainY, testY) = train_test_split(
# 	train, target , test_size = 0.33)

# train the Deep Belief Network with 784 input units (the flattened,
#  28x28 grayscale image), 800 hidden units in the 1st hidden layer,
# 800 hidden nodes in the 2nd hidden layer, and 10 output units (one
# for each possible output classification, which are the digits 1-10)
dbn = DBN(
	[train.shape[1], 800, 800, 10],
	learn_rates = 0.3,
	learn_rate_decays = 0.9,
	epochs = 15,
	verbose = 1)
dbn.fit(train, target)

# compute the predictions for the test data and show a classification
# report

pred = dbn.predict(test)

np.savetxt('submission_dbnn.csv', np.c_[range(1,len(test)+1),pred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')

# print metrics.classification_report(testY, preds)

# print metrics.accuracy_score(testY, preds)