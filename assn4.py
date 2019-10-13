# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 21:21:49 2019

@author: khris
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
NameList = ["sepal length", "sepal width", "petal length", "petal width", "class"]
newData = pd.read_csv("iris.data", names = NameList, header = None)

#question 4
print(newData.head())
sns.lmplot(x = "sepal width", y = "sepal length", hue = "class" ,data = newData)
sns.lmplot(x = "petal width", y = "petal length", hue = "class", data = newData)


le = preprocessing.LabelEncoder()

#question 5
features = list(zip(newData["sepal length"], newData["sepal width"], newData["petal length"], newData["petal width"]))
#the response is the class of data
Y = newData["class"]
Y = Y.tolist()


#partition data into train, test and split
X_train, X_test, Y_train, Y_test = train_test_split(features, Y, test_size = 0.3)

#create the classifier and run it on the data set
j = KNeighborsClassifier()
j.fit(X_train, Y_train)


#function that takes in two arrays, the predicted Y values and the Actual Y values
#output is the percentage of values predicted correctly
def correctCount(predictedArray, actualArray):
    count = 0
    for i in range(0, len(actualArray)):
        #if item class we predicted matches actual class add one to the count
        if(predictedArray[i] == actualArray[i]):
            count += 1
    return count
    
#predict the class for the test set   
#display it to the user
#this is question 8
predictionArray = j.predict(X_test)
totalCorrect = correctCount(predictionArray, Y_test)
totalIncorrect = (len(Y_test) - totalCorrect)
accuracy = (totalCorrect / (len(Y_test))) * 100
print("Total predictions correct: ", totalCorrect)
print("Total predictions incorrect: ", totalIncorrect)
print("Accuracy with % Neighbours: %", accuracy)


#question 9
numberOfNeighbors = [1,3, 5, 7, 10,20, 30, 40, 50]
accuracyList = []
totalCorrectList = []

#for each number
for x in numberOfNeighbors:
    total = 0
    totalRightPredictions = 0
    #loop thru and do ten random test/train sets
    for y in range(0, 10):
        x_train, x_test, y_train, y_test = train_test_split(features, Y, test_size = 0.3)
        classifier = KNeighborsClassifier(n_neighbors = x)
        classifier.fit(x_train, y_train)
        predictionArray = classifier.predict(x_test)
        totalCorrect = correctCount(predictionArray, y_test)
        totalIncorrect = len(y_test) - totalCorrect
        accuracy = (totalCorrect / len(y_test)) * 100
        total = total + accuracy
        totalRightPredictions = totalRightPredictions + totalCorrect
    #now divide accuracy by ten to get the average accuracy
    average = total / 10
    totalRightPredictions = round(totalRightPredictions / 10)
    #add it to the list
    accuracyList.append(average)
    totalCorrectList.append(totalRightPredictions)

plt.figure()    
plt.scatter(numberOfNeighbors, accuracyList)
plt.xlabel("Number of Neighbors")
plt.ylabel("Prediction Accuracy (%)")
plt.title("plotting accuracy of KNN for various numbers of K")

#it appears as though when going from 10 neighbors to 20 neighbors, accuracy dramatically drops
#I reran the code several times, and each time it appears that K values less than or 
#equal to ten return the most accurate classifications.