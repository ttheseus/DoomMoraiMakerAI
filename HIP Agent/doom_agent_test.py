'''
Agent Training Co-Creative Machine Learning
'''

import pickle
import csv, random, glob, random

import numpy as np
import tflearn
import tensorflow as tf
from tflearn import conv_2d, conv_3d, max_pool_2d, local_response_normalization, batch_normalization, fully_connected, \
    regression, input_data, dropout, custom_layer, flatten, reshape, embedding, conv_2d_transpose

import copy
import sys

np.random.seed(1)
random.seed(1)
tf.random.set_seed(1)

characters = ["-", "X", ".", ",", "E", "W", "A", "H", "B", "K", "<", "T", ":", "L", "t", "+", ">"]

testX = pickle.load(open("doom_testX.p", "rb"))#40 (width) x 15 (height) x 34 (SMB entitites) (state)
testY = pickle.load(open("doom_testY.p", "rb"))#40 (width) x 15 (height) x 32 (SMB entities except the player and the flat) (value of the AI making a particular addition)

testX = np.reshape(testX, (-1, 14, 14, 17))
testY = np.reshape(testY, (-1, 14, 14, 17))

print (testX.shape)
print (testY.shape)

#Architecture
networkInput = tflearn.input_data(shape=[None, 14, 14, 17])
conv = conv_2d(networkInput, 8,4, activation='leaky_relu')
conv2 = conv_2d(conv, 16,3, activation='leaky_relu')
conv3 = conv_2d(conv2, 32,3, activation='leaky_relu')
fc = tflearn.fully_connected(conv3, 14*14*17, activation='leaky_relu')
mapShape = tf.reshape(fc, [-1,14,14,17])
network = tflearn.regression(mapShape, optimizer='adam', metric='R2', loss='mean_square',learning_rate=0.0001)##

model = tflearn.DNN(network)
model.load("doom_test_model.tflearn")
testYPrime = model.predict(testX)

#Visualize
for j in range(0, len(testYPrime)):
	pred = testYPrime[j]
	chunk = ""
	for y in range(0, 14):
		line = ""
		for x in range(0, 14):
			maxIndex = -1
			maxValue = -1
			for i in range(0, len(pred[y][x])):
				if pred[y][x][i]>maxValue:
					maxValue = pred[y][x][i]
					maxIndex = i
			if maxIndex==-1:
				line+= " "
			else:
				line += characters[maxIndex]
		line += "\r\n"
		chunk += line
	print("New Actions Predicted") 
	print(chunk)

	trueNext = testY[j]

	chunkTrue = ""
	for y in range(0, 14):
		line = ""
		for x in range(0, 14):
			maxIndex = -1
			maxValue = -1
			if sum(trueNext[y][x])==0:
				line += " "
			else:
				for i in range(0, len(trueNext[y][x])):
					if trueNext[y][x][i]>maxValue:
						maxValue = trueNext[y][x][i]
						maxIndex = i

				line += characters[maxIndex]
		line += "\r\n"
		chunkTrue += line
	print("True Action") 
	print(chunkTrue)

	trueState = testX[j]

	chunkTrue = ""
	for y in range(0, 14):
		line = ""
		for x in range(0, 14):
			maxIndex = -1
			maxValue = -1
			for i in range(0, len(trueState[y][x])):
				if trueState[y][x][i]>maxValue:
					maxValue = trueState[y][x][i]
					maxIndex = i

			line += characters[maxIndex]
		line += "\r\n"
		chunkTrue += line
	print("True State") 
	print(chunkTrue)