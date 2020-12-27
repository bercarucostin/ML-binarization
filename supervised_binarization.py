import csv
import json
import statistics
import numpy as np
import pandas as pd
import os
import cv2
import tensorflow as tf
import random

from tqdm import tqdm
from math import ceil
from keras.utils import Sequence
from keras.models import Sequential, load_model
from keras.regularizers import l1, l2, l1_l2
from keras.callbacks import ModelCheckpoint
from keras.layers import Activation
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from keras import backend as K

useCols = [29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57 ,58 ,59, 60, 61, 62, 63, 64, 65, 66]

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
	y_pred = K.round(y_pred)
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

#0.001 L2 (0.86834) , 0.0001 L2 (0.90957) ca activity a mers bine
#0.01 l1 (0.923668), 0.01 l2 (0.920261) ca kernel a mers bine
#0.001 l2 0.9221938838355543 ca bias
#0.01 l1 kernel + 0.001 l2 bias 0.9226663780366464 
#0.01 l2 kernel + 0.001 l2 bias 0.9135232308930193 
def createModel():
	model = Sequential()
	model.add(Dense(256, input_dim=len(useCols), activation='relu'))
	# model.add(Dropout(0.3))

	# if reg == '1':
	# 	model.add(Dense(256, activation='linear', bias_regularizer=l2(0.001), kernel_regularizer=l1(0.01)))
	# elif reg == '2':
	# 	model.add(Dense(256, activation='linear', bias_regularizer=l2(0.001), kernel_regularizer=l2(0.01)))
	# model.add(Activation('relu'))

	
	model.add(Dense(128, activation='linear', bias_regularizer=l2(0.001), kernel_regularizer=l2(0.01)))
	model.add(Activation('relu'))

	model.add(Dense(64, activation='linear', bias_regularizer=l2(0.001), kernel_regularizer=l2(0.01)))
	model.add(Activation('relu'))

	model.add(Dense(32, activation='linear', bias_regularizer=l2(0.001), kernel_regularizer=l2(0.01)))
	model.add(Activation('relu'))

	model.add(Dense(16, activation='linear', bias_regularizer=l2(0.001), kernel_regularizer=l2(0.01)))
	model.add(Activation('relu'))

	model.add(Dropout(0.3)) 
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss=f1_loss, optimizer='adam', metrics=['accuracy', f1, precision_m, recall_m])
	return model

def readTrainData():
	pathToData = 'data'
	trainDataX = []
	trainDataY = []

	for f in tqdm(os.listdir('dataTrain')):
		if 'OUT' in f:
			trainDataY.append(pd.read_csv('dataTrain/' +  f, header=None))
			trainDataX.append(pd.read_csv('dataTrain/' + f.replace('OUT', 'IN'), header=None, usecols=useCols))        
			
	pdTrainData_X = trainDataX[0]
	pdTrainData_Y = trainDataY[0]

	# del trainDataX[0]
	# del trainDataY[0]

	for i in tqdm(range(1, len(trainDataX))):
		pdTrainData_X = pdTrainData_X.append(trainDataX[i])
		pdTrainData_Y = pdTrainData_Y.append(trainDataY[i])
		# del trainDataX[0]
		# del trainDataY[0]
		
	del trainDataX
	del trainDataY

	ss = StandardScaler()
	trainData_X = ss.fit_transform(pdTrainData_X.to_numpy())
	trainData_Y = pdTrainData_Y.to_numpy()
	del pdTrainData_X
	del pdTrainData_Y

	trainX, valX, trainY, valY = train_test_split(trainData_X, trainData_Y, test_size = 0.05, random_state = 42)
	trainY= trainY.ravel()
	n_timesteps, n_features, n_outputs = trainX.shape[0], trainX.shape[1], trainY.shape[0]
	del trainData_X
	del trainData_Y
	return trainX, trainY, valX, valY, ss, n_timesteps, n_features, n_outputs

def readTestData():
	gtFiles = os.listdir('dataGT')
	sizes = []
	testDataX = []
	testDataY = []
	for f in tqdm(os.listdir('dataTest')):
		if 'OUT' in f:
			print('dataTest/' +  f)
			print('dataTest/' +  f.replace('OUT', 'IN'))
			testDataY.append(pd.read_csv('dataTest/' +  f, header=None))
			testDataX.append(pd.read_csv('dataTest/' +  f.replace('OUT', 'IN'), header=None, usecols=useCols))
			for gt in gtFiles:
				if f.split('.')[0] in gt:
					print('dataGT/' + gt)
					x, y, _ = cv2.imread('dataGT/' + gt).shape
					sizes.append((x, y))
					break
			
	for i in range(0, len(testDataX)):
		testDataX[i] = ss.transform(testDataX[i].to_numpy())
		testDataY[i] = testDataY[i].to_numpy()
	
	return testDataX, testDataY, sizes

trainX, trainY, valX, valY, ss, n_timesteps, n_features, n_outputs = readTrainData()

unique, counts = np.unique(trainY, return_counts=True)
nrExamples = (dict(zip(unique, counts)))
print(nrExamples[0] / nrExamples[1])
class_weight = {0: 1.,
				1: nrExamples[0] / nrExamples[1]}


model = createModel(reg)
model.fit(trainX, trainY, epochs=7, batch_size=1000, class_weight=class_weight, validation_data=(valX, valY), verbose=1)

del trainX
del trainY
del valX
del valY


fmAvg = 0.0


gtFiles = os.listdir('dataGT')
sizes = None
testDataX = None
testDataY = None
for f in tqdm(os.listdir('dataTest')):
	if 'OUT' in f:
		print('dataTest/' +  f)
		print('dataTest/' +  f.replace('OUT', 'IN'))
		testDataY = pd.read_csv('dataTest/' +  f, header=None).to_numpy()
		testDataX = ss.transform(pd.read_csv('dataTest/' +  f.replace('OUT', 'IN'), header=None, usecols=useCols).to_numpy())
		# for gt in gtFiles:
		# 	if f.split('.')[0] in gt:
		# 		print('dataGT/' + gt)
		# 		x, y, _ = cv2.imread('dataGT/' + gt).shape
		# 		sizes = ((x, y))
		# 		break
		nnPredTest = model.predict(testDataX)
		print("NN accuracies and F1:")
		print(accuracy_score(testDataY, np.where(nnPredTest > 0.5, 1, 0)))
		score = f1_score(testDataY, np.where(nnPredTest > 0.5, 1, 0))
		fmAvg += score
		print(score)
		# print("")
		# print(confusion_matrix(testDataY, np.where(nnPredTest > 0.5, 1, 0)))
		# xy = sizes
		# img = np.vectorize(turnToValidImg)(np.where(nnPredTest > 0.5, 1, 0))
		# img = np.resize(img, xy)
		# cv2.imwrite('dataGT/NN' + str(reg) + str(i) + '_' + str(score) + '_.tif', img)
print("Final one")
print(fmAvg / len(testX), reg)

for i in range(0, len(testDataX)):
	testDataX[i] = ss.transform(testDataX[i].to_numpy())
	testDataY[i] = testDataY[i].to_numpy()



print("")
print("")
for i in range(0, len(testX)):
	nnPredTest = model.predict(testX[i])
	print("NN accuracies and F1:")
	print(accuracy_score(testY[i], np.where(nnPredTest > 0.5, 1, 0)))
	score = f1_score(testY[i], np.where(nnPredTest > 0.5, 1, 0))
	fmAvg += score
	print(score)
	print("")
	print(confusion_matrix(testY[i], np.where(nnPredTest > 0.5, 1, 0)))
	xy = sizes[i]
	img = np.vectorize(turnToValidImg)(np.where(nnPredTest > 0.5, 1, 0))
	img = np.resize(img, xy)
	cv2.imwrite('dataGT/NN' + str(reg) + str(i) + '_' + str(score) + '_.tif', img)
print("Final one")
print(fmAvg / len(testX), reg)