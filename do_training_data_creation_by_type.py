# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import glob, math
import pickle 
import random

levels = []#list of dictionaries, each dictionary a level

characters = ["-", "X", ".", ",", "E", "W", "A", "H", "B", "K", "<", "T", ":", "L", "t", "+", ">"]

#Load Doom Levels
for levelFile in glob.glob(".\\Doom_Level_Data\\*.txt"): # *.txt for all, change to test later
  print ("Processing: "+levelFile)
  with open(levelFile) as fp:
    level = {}
    y = 0
    for line in fp:
      level[y] = {}
      for x in range(0, len(line)-1):
        onehot = [0]*len(characters)
        onehot[characters.index(line[x])] =1
        level[y][x]= onehot
      y+=1
    levels.append(level)


X = []
Y = []
for level in levels:
    for x in range(0, len(level[0].keys())-14, 7):
        for y in range(0, len(level.keys())-14, 7):
            chunk = []
            onlyEmpty = True
            for yi in range(0, 14):
                line = []
                for xi in range(0, 14):
                    line.append(level[y+yi][x+xi])
                    if level[y+yi][x+xi][0] == 0:
                        onlyEmpty = False
                chunk.append(line)
            if not onlyEmpty:
                X.append(chunk)
                Y.append(chunk)


newXTrainingData = []
newYTrainingData = []
#Tilebased Reprocessing

index = 0
for x in X:
    print (str(index)+" of "+str(len(X)))
    index+=1
    currX = list(x)
    desiredPosition = 0
    while desiredPosition < 17: 
        #cloneX = list(x)
        everRemoveAnything = False

        cloneX = []
        for y in range(0, 14):
            line = []
            for xi in range(0, 14):
                line.append(currX[y][xi])
            cloneX.append(line)
        
        
        additionsY = []
        for y in range(0, 14):
            line = []
            for xi in range(0, 14):
                line.append([0]*17)
            additionsY.append(line)
        
        for y in range (0, 14):
            for xi in range(0, 14):
                if 1 in cloneX[y][xi] and cloneX[y][xi].index(1) == desiredPosition:
                    everRemoveAnything = True
                    additionsY[y][xi] = list(cloneX[y][xi])
                    cloneX[y][xi] = [0]*len(characters)
        if everRemoveAnything:
            newXTrainingData.append(cloneX)
            littleY = np.array(additionsY)
            newYTrainingData.append(littleY)
            currX = cloneX
            

        desiredPosition +=1
X = np.array(newXTrainingData)
#print(len(newYTrainingData))
Y = np.array(newYTrainingData)

pickle.dump(X, open("doom_trainX.p", "wb"))
pickle.dump(Y, open("doom_trainY.p", "wb")) # change to doom_test later for test data, run doom_agent with train, run doom_agent_test with test

print ("Y Shape: "+str(Y.shape))

#Visualize State
for j in range(0, len(newXTrainingData)):
    chunkTrue = ""
    for y in range(0, 14):
        line = ""
        for x in range(0, 14):
            maxIndex = -1
            maxValue = 0
            if sum(newXTrainingData[j][y][x])==0:
                line += " "
            else:
                for i in range(0, len(newXTrainingData[j][y][x])):
                    if newXTrainingData[j][y][x][i]>maxValue:
                        maxValue = newXTrainingData[j][y][x][i]
                        maxIndex = i
                line += characters[maxIndex]
        line +="\n"
        chunkTrue+=line

    print("True State")
    print(chunkTrue)
    chunkTrue = ""
    for y in range(0, 14):
        line = ""
        for x in range(0, 14):
            maxIndex = -1
            maxValue = 0
            if sum(newYTrainingData[j][y][x])==0:
                line += " "
            else:
                for i in range(0, len(newYTrainingData[j][y][x])):
                    if newYTrainingData[j][y][x][i]>maxValue:
                        maxValue = newYTrainingData[j][y][x][i]
                        maxIndex = i
                line += characters[maxIndex]
        line +="\n"
        chunkTrue+=line

    print("True Action")
    print(chunkTrue)