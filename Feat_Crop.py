
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2, scipy


def crop_matrix(matrix):

	for index, x in np.ndenumerate(matrix):
		if x != 0:
			a = index[0]
			break

	for index, x in np.ndenumerate(matrix.T):
		if x != 0:
			b = index[0]
			break

	cropped_matrix = matrix[a:a+20,b:b+20]

	wall = np.zeros((20,20),dtype=np.int)

	wall[0:0+cropped_matrix.shape[0], 0:0+cropped_matrix.shape[1]] = cropped_matrix

	return wall

SZ=28
affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR
def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
    return img

def compareImages(original, modified1, modified2, modified3):
	# plot the progress
	ax1 = plt.subplot(1,4,1)
	ax2 = plt.subplot(1,4,2)
	ax3 = plt.subplot(1,4,3)
	ax4 = plt.subplot(1,4,4)
	ax1.imshow(original, cmap=cm.binary)
	ax1.set_title('Example1')
	ax2.imshow(modified1, cmap=cm.binary)
	ax2.set_title('Example2')
	ax3.imshow(modified2, cmap=cm.binary)
	ax3.set_title('E1 Deskewed')
	ax4.imshow(modified3, cmap=cm.binary)
	ax4.set_title('E2 Deskewed')
	plt.show()

def compare2Images(original,modified1):
	ax1 = plt.subplot(2,2,1)
	ax2 = plt.subplot(2,2,2)
	ax1.imshow(original, cmap=cm.binary)
	ax1.set_title('1')
	ax2.imshow(modified1, cmap=cm.binary)
	ax2.set_title('2')
	plt.show()

def compare3Images(original,modified1, modified2):
	ax1 = plt.subplot(1,3,1)
	ax2 = plt.subplot(1,3,2)
	ax3 = plt.subplot(1,3,3)
	ax1.imshow(original, cmap=cm.binary)
	ax1.set_title('Original')
	ax2.imshow(modified1, cmap=cm.binary)
	ax2.set_title('Cropped')
	ax3.imshow(modified2, cmap=cm.binary)
	ax3.set_title('Deskewed')
	plt.show()

dataset = pd.read_csv("../data/train.csv")
target = dataset[[0]].values.ravel()
train = dataset.iloc[:,1:].values
test = pd.read_csv("../data/test.csv").values

# 4 - 143,42

train_cropped = []
train_desk = []
train_cropped_desk = []

# compareImages(train[42].reshape((28, 28)).astype(np.uint8),train[143].reshape((28, 28)).astype(np.uint8), 
# 	deskew(train[42].reshape((28, 28)).astype(np.uint8)) ,deskew(train[143].reshape((28, 28)).astype(np.uint8)))

# with open('features/vanilla.npy', 'wb') as f:
# 	np.save(f, np.array([train, test]))

# exit()

print "Preprocessing train features"
for feature in train:
	matrix = feature.reshape((28, 28)).astype(np.uint8)

	deskewed = deskew(matrix)

	cropped = crop_matrix(matrix)

	cropped_deskewed = crop_matrix(deskewed)

	train_cropped_desk.append(cropped_deskewed.ravel())
	train_cropped.append(cropped.ravel())
	train_desk.append(deskewed.ravel())

	# print cropped.shape
	# compareImages(matrix, cropped, deskewed, cropped_deskewed)

test_cropped = []
test_desk = []
test_cropped_desk = []

print "Preprocessing test features"
for feature in test:
	matrix = feature.reshape((28, 28)).astype(np.uint8)

	deskewed = deskew(matrix)

	cropped = crop_matrix(matrix)

	cropped_deskewed = crop_matrix(deskewed)

	test_cropped_desk.append(cropped_deskewed.ravel())
	test_cropped.append(cropped.ravel())
	test_desk.append(deskewed.ravel())

cropped = np.array([train_cropped, test_cropped])
skewed = np.array([train_desk, test_desk])
cropped_skewed = np.array([train_cropped_desk, test_cropped_desk])

# print "Saving to *.npy"
# with open('features/cropped.npy', 'wb') as f:
# 	np.save(f, cropped)

with open('features/deskewed.npy', 'wb') as f:
	np.save(f, skewed)

# with open('features/cropped_deskewed.npy', 'wb') as f:
# 	np.save(f, cropped_skewed)



