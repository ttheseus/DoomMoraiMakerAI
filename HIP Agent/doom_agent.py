'''
Agent Training Co-Creative Machine Learning
'''

import pickle
import csv, random, glob, random

import numpy as np
import tflearn
import tensorflow as tf
from tflearn import conv_2d, fully_connected, \
    regression, input_data, custom_layer, flatten, reshape, embedding

import copy
import sys

np.random.seed(1)
random.seed(1)
tf.random.set_seed(1)

characters = ["-", "X", ".", ",", "E", "W", "A", "H", "B", "K", "<", "T", ":", "L", "t", "+", ">"]

trainX = pickle.load(open("doom_trainX.p", "rb"))#40 (width) x 15 (height) x 34 (SMB entitites) (state)
trainY = pickle.load(open("doom_trainY.p", "rb"))#40 (width) x 15 (height) x 32 (SMB entities except the player and the flat) (value of the AI making a particular addition)

trainX = np.array(trainX)
trainY = np.array(trainY)
trainX = np.reshape(trainX, (-1, 14, 14, 17))
trainY = np.reshape(trainY, (-1, 14, 14, 17))

print (trainX.shape)
print (trainY.shape)

namesToLoad = ["convW", "convb", "conv2W", "conv2b", "conv3W", "conv3b", "fcW", "fcb"]
layers = {}

for name in namesToLoad:
	layers[name] = pickle.load(open(name+".p", "rb"))
	print(layers[name].shape)

#Architecture
networkInput = tflearn.input_data(shape=[None, 14, 14, 17])
conv = conv_2d(networkInput, 8,4, activation='leaky_relu')
conv2 = conv_2d(conv, 16,3, activation='leaky_relu')
conv3 = conv_2d(conv2, 32,3, activation='leaky_relu')
fc = tflearn.fully_connected(conv3, 14*14*17, activation='leaky_relu')
mapShape = tf.reshape(fc, [-1, 14, 14, 17])
network = tflearn.regression(mapShape, optimizer='adam', metric='R2', loss='mean_square',learning_rate=0.0001)##

model = tflearn.DNN(network)

newDoomWeightValues = np.zeros(conv.W.shape)

for a in range(0,4):
	for b in range(0,4):
		for c in range(0,17):
			for d in range(0,8):
				newDoomWeightValues[a,b,c,d] = layers["convW"][a,b,c,d]

model.set_weights(conv.W, newDoomWeightValues)

'''

weight = pickle.load("convW.p", "rb")

model.set_weights(conv.W, weight)
fileNameToLoadedValue[key] = weights
#model.load("ki_test_model.tflearn")
'''

model.fit(trainX, 
 		Y_targets=trainX, 
 		n_epoch=20,
 		shuffle=True, 
 		show_metric=True, 
 		snapshot_epoch=False,
 		batch_size=20,
 	    run_id='cocreativeTest')

model.save("doom_test_model.tflearn")
