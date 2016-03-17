from sklearn.ensemble import RandomForestClassifier


from sklearn.cross_validation import cross_val_score
import numpy as np
import pandas as pd

with open('cropped_deskewed.npy', 'rb') as f:
	features = np.load(f)

train = np.asarray(features[1])
target = np.asarray(features[0])

rf = RandomForestClassifier(n_estimators=2500,n_jobs = -1)

print "Running RFC"

cut = int(len(train) / 10)

rf.fit(train[:len(train) - cut],target[:len(train) - cut])

print rf.score(train[-cut:],target[-cut:])


# scores = cross_val_score(rf, train, target, cv=3)
# print scores

# rf.fit(train, target)
# pred = rf.predict(test)

# np.savetxt('submission_rand_forest.csv', np.c_[range(1,len(test)+1),pred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')