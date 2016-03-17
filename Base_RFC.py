from sklearn.ensemble import RandomForestClassifier


from sklearn.cross_validation import cross_val_score
import numpy as np
import pandas as pd

# create the training & test sets, skipping the header row with [1:]
dataset = pd.read_csv("../data/train.csv")
target = dataset[[0]].values.ravel()
train = dataset.iloc[:,1:].values
test = pd.read_csv("../data/test.csv").values

# create and train the random forest
# multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
rf = RandomForestClassifier(n_estimators=100,n_jobs = -1)

scores = cross_val_score(OneVsRestClassifier(rf), train, target, cv=5)


print scores

# rf.fit(train, target)
# pred = rf.predict(test)

# np.savetxt('submission_rand_forest.csv', np.c_[range(1,len(test)+1),pred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')