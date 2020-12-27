import csv
import json
import statistics
import numpy as np
import pandas as pd
import os
import cv2
# import tensorflow as tf
import random

from tqdm import tqdm
from math import ceil
# from keras.constraints import max_norm
# from keras.utils import Sequence
# from keras.models import Sequential, load_model, Model
# from keras.regularizers import l1, l2, l1_l2
# from keras.callbacks import ModelCheckpoint
# from keras.layers import Activation
# from keras.layers import Dense, Dropout, Input, BatchNormalization, ZeroPadding1D
# from keras.layers import Flatten
# from keras.layers import Add, Activation, Conv1D, AveragePooling1D, MaxPooling1D, GlobalMaxPooling1D
# from keras.layers.convolutional import Conv1D
# from keras.layers.convolutional import MaxPooling1D
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
# from keras.initializers import glorot_uniform
# from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam

from sklearn.cluster import KMeans, SpectralBiclustering, SpectralCoclustering, AffinityPropagation, AgglomerativeClustering, DBSCAN, Birch, FeatureAgglomeration, MiniBatchKMeans, MeanShift

# from keras import backend as K

useCols = [29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57 ,58 ,59, 60, 61, 62, 63, 64, 65, 66]
# without neighbours intesities
useCols1 = [12, 13, 14, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57 ,58 ,59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88 ,89, 90, 91, 92, 93]
# without neighbours intesities and progressive pixel centered window statistics
useCols2 = [12, 13, 14, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57 ,58 ,59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88 ,89, 90, 91, 92, 93]
# without neighbours intesities and progressive pixel centered window statistics and locality statistics
useCols4 = [0, 1, 2, 3, 4, 5, 6, 7 ,8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57 ,58 ,59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81]

EPS1 = 0.15
EPS2 = 0.05

def recall_m(y_true, y_pred):
		true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
		possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
		recall = true_positives / (possible_positives + K.epsilon())
		return recall

def precision_m(y_true, y_pred):
		true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
		predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
		precision = true_positives / (predicted_positives + K.epsilon())
		return precision

def f1(y_true, y_pred):
	# y_pred = K.round(y_pred)
	tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
	tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
	fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
	fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

	p = tp / (tp + fp + K.epsilon())
	r = tp / (tp + fn + K.epsilon())

	f1 = 2*p*r / (p+r+K.epsilon())
	f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
	return K.mean(f1)

def f1_loss(y_true, y_pred):

	tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
	tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
	fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
	fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

	p = tp / (tp + fp + K.epsilon())
	r = tp / (tp + fn + K.epsilon())

	f1 = 2*p*r / (p+r+K.epsilon())
	f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
	return 1 - K.mean(f1)


def f1_m(y_true, y_pred):
	precision = precision_m(y_true, y_pred)
	recall = recall_m(y_true, y_pred)
	return 2*((precision*recall)/(precision+recall+K.epsilon()))

def turnToValidImg(x):
	if (x == 0):
		x = 255
	else:
		x = 0
	return x

# def readTrainData():

sizes = []
pathToData = 'data'
trainDataX = []
trainDataY = []

for f in tqdm(os.listdir('dataTrainKMeans')):
	# if 'OUT' in f:
	# 	x, y, _ = cv2.imread('GTKMeans/GT_' + f.replace('.OUT.CSV', '')).shape
	# 	sizes = [(x, y)]
	# 	trainDataY.append(pd.read_csv('dataTrainKMeans/' +  f, header=None))
	# 	trainDataX.append(pd.read_csv('dataTrainKMeans/' + f.replace('OUT', 'IN'), header=None))
	# 	# trainDataX.append(pd.read_csv('dataTrainKMeans/' + f.replace('OUT', 'IN'), header=None))
	# else:
	# 	continue
	print("Reading " + f)
	x, y, _ = cv2.imread('GTKMeans/GT_' + f.replace('CSV', 'tiff')).shape
	sizes = [(x, y)]
	trainDataX.append(pd.read_csv('dataTrainKMeans/' + f, header=None))



	pdTrainData_X = trainDataX[0]
	# pdTrainData_Y = trainDataY[0]

	for i in tqdm(range(1, len(trainDataX))):
		pdTrainData_X = pdTrainData_X.append(trainDataX[i])
		# pdTrainData_Y = pdTrainData_Y.append(trainDataY[i])

	del trainDataX
	# del trainDataY
	trainDataX = []
	# trainDataY = []

	ss = StandardScaler()
	X = ss.fit_transform(pdTrainData_X.to_numpy())
	# Y = pdTrainData_Y.to_numpy()

	spectralbi = SpectralBiclustering(n_clusters=2, n_components=20, n_best=10, method='log', svd_method='arpack', random_state=42, n_jobs=-1)
	spectralbi_notarpack = SpectralBiclustering(n_clusters=2, n_components=20, n_best=10, method='log', random_state=42, n_jobs=-1)
	spectralScale = SpectralBiclustering(n_clusters=2, n_components=20, n_best=10, method='scale', svd_method='arpack', random_state=42, n_jobs=-1)
	spectralScale_notaprack = SpectralBiclustering(n_clusters=2, n_components=20, n_best=10, method='scale', svd_method='arpack', random_state=42, n_jobs=-1)
	spectralBistochastic = SpectralBiclustering(n_clusters=2, n_components=20, n_best=10, svd_method='arpack', random_state=42, n_jobs=-1)
	spectralBistochastic_notarpack = SpectralBiclustering(n_clusters=2, n_components=20, n_best=10, random_state=42, n_jobs=-1)

	alternatives = [spectralbi_notarpack, spectralScale, spectralScale_notaprack, spectralBistochastic, spectralBistochastic_notarpack]

	spectralbi.fit(X)

	nr_of_1 = sum(spectralbi.row_labels_)
	nr_of_0 = len(spectralbi.row_labels_) - nr_of_1

	whosBigger = nr_of_0/nr_of_1
	if whosBigger < 1:
		spectralbi.row_labels_ = list(map(lambda x: 1 - x, spectralbi.row_labels_))

	nr_of_1 = sum(spectralbi.row_labels_)
	nr_of_0 = len(spectralbi.row_labels_) - nr_of_1

	percentage_of_0 = nr_of_0/(nr_of_0+nr_of_1)
	percentage_of_1 = nr_of_1/(nr_of_1+nr_of_0)
	
	print(percentage_of_0, percentage_of_1)

	cnt = -1
	while ((percentage_of_0 - percentage_of_1 < EPS1 or 1 - percentage_of_0 < EPS2) and cnt < 4):
		cnt += 1	
		alternatives[cnt].fit(X)
		spectralbi.row_labels_ = alternatives[cnt].row_labels_

		nr_of_1 = sum(spectralbi.row_labels_)
		nr_of_0 = len(spectralbi.row_labels_) - nr_of_1

		percentage_of_0 = nr_of_0/(nr_of_0+nr_of_1)
		percentage_of_1 = nr_of_1/(nr_of_1+nr_of_0)

		print(cnt, percentage_of_0, percentage_of_1)

	whosBigger = nr_of_0/nr_of_1
	if whosBigger < 1:
		spectralbi.row_labels_ = list(map(lambda x: 1 - x, spectralbi.row_labels_))

	xy = sizes[0]
	img1 = np.vectorize(turnToValidImg)(spectralbi.row_labels_)
	img = np.resize(img1, xy)
	cv2.imwrite(f.replace('CSV', 'tiff'), img)

	# xy = sizes[0]
	# img = np.vectorize(turnToValidImg)(feaagg.labels_)
	# img = np.resize(img, xy)
	# cv2.imwrite('dataTrainKMeans/' + f.replace('.OUT.CSV', '') + 'TEHSHIT5_.tif', img)

	# xy = sizes[0]
	# img = np.vectorize(turnToValidImg)(minikmeans.labels_)
	# img = np.resize(img, xy)
	# cv2.imwrite('dataTrainKMeans/' + f.replace('.OUT.CSV', '') + 'TEHSHIT6_.tif', img)


# fmAvg = 0.0
# fmAvg1 = 0.0
# fmAvg2 = 0.0
# cnt = 0



# 		print("NN accuracies and F1:")
# 		print(accuracy_score(testDataY, np.where(nnPredTest > 0.5, 1, 0)))
# 		score = f1_score(testDataY, np.where(nnPredTest > 0.5, 1, 0))
# 		fmAvg += score
# 		print(score)
# 		print(accuracy_score(testDataY, np.where(nnPredTest1 > 0.5, 1, 0)))
# 		score1 = f1_score(testDataY, np.where(nnPredTest1 > 0.5, 1, 0))
# 		fmAvg1 += score1
# 		print(score1)
# 		# print(accuracy_score(testDataY, np.where(nnPredTest2 > 0.5, 1, 0)))
# 		# score2 = f1_score(testDataY, np.where(nnPredTest2 > 0.5, 1, 0))
# 		# fmAvg2 += score2
# 		# print(score2)
# 		cnt += 1
# 		# print("")
# 		# print(confusion_matrix(testDataY, np.where(nnPredTest > 0.5, 1, 0)))
# 		# xy = sizes
# 		# img = np.vectorize(turnToValidImg)(np.where(nnPredTest > 0.5, 1, 0))
# 		# img = np.resize(img, xy)
# 		# cv2.imwrite('dataGT/NN' + str(reg) + str(i) + '_' + str(score) + '_.tif', img)
# print("Final one")
# print(fmAvg / cnt)
# print(fmAvg1 / cnt)

# for i in range(0, len(testDataX)):
# 	testDataX[i] = ss.transform(testDataX[i].to_numpy())
# 	testDataY[i] = testDataY[i].to_numpy()

# print("")
# print("")
# for i in range(0, len(testX)):
# 	nnPredTest = model.predict(testX[i])
# 	print("NN accuracies and F1:")
# 	print(accuracy_score(testY[i], np.where(nnPredTest > 0.5, 1, 0)))
# 	score = f1_score(testY[i], np.where(nnPredTest > 0.5, 1, 0))
# 	fmAvg += score
# 	print(score)
# 	print("")
# 	print(confusion_matrix(testY[i], np.where(nnPredTest > 0.5, 1, 0)))
# 	xy = sizes[i]
# 	img = np.vectorize(turnToValidImg)(np.where(nnPredTest > 0.5, 1, 0))
# 	img = np.resize(img, xy)
# 	cv2.imwrite('dataGT/NN' + str(reg) + str(i) + '_' + str(score) + '_.tif', img)
# print("Final one")
# print(fmAvg / len(testX), reg)