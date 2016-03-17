# Import the modules
from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
import cv2
import os, scipy
import numpy as np
import pandas as pd


def performRecognition(matrix):

	# Read the input image 
	im = cv2.imdecode(matrix.reshape(28,28),0)


	# Convert to grayscale and apply Gaussian filtering
	im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

	# Threshold the image
	ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

	# Find contours in the image
	ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# Get rectangles contains each contour
	rects = [cv2.boundingRect(ctr) for ctr in ctrs]

	for rect in rects:
		# Draw the rectangles
		cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
		# Make the rectangular region around the digit
		leng = int(rect[3] * 1.6)
		pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
		pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
		roi = im_th[pt1:pt1+leng, pt2:pt2+leng]

		roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
		roi = cv2.dilate(roi, (3, 3))

		roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)

		print roi_hog_fd



def vnToHog():
	with open('features/vanilla.npy', 'rb') as f:
		features = np.load(f)

	features
	train = []

	print 'Hogging '

	print 'Train data'

	for feature in features[0]:

		performRecognition(feature)
		return
		train.append(fd)

	train = np.array(train, 'float64')

	print 'Test data'
	test = []
	for feature in features[1]:
		test.append(fd)

	test = np.array(test, 'float64')

	print 'Saving features'
	data = np.array([train, test])

	with open('features/vn_toHoG' , 'wb') as f:
		np.save(f, data)

vnToHog()