# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 13:23:02 2017

@author: Hubert
"""

import InputOutput as io
import dataWrangling as dW
import xgboostTraining as xgbt

import numpy as np
import pandas as pd

import datetime

from xgboost import plot_importance
from matplotlib import pyplot

from sklearn.model_selection import train_test_split


#Settings used to save time if you need to
mode = 4 #used to determine whih dataset we're reading from
read = 1 #change to 1 if you want to re-read dataframes, otherwise, set this to 0
dataWrangle = 1 #change to 1 if you want to rewrangle data
verify = 0 #change to 1 if you want to remake the verification sets
model = 1


if(read):
    currentTime = datetime.datetime.now().isoformat()
    print("Reading time begin:", currentTime)
    
    print("Reading Training Database, mode {} ...".format(mode))
    #trainDF = io.readTrainData(mode)
    trainDF = io.lesReadTrainData(mode)
    
    print("Reading Test Database...")
    testDF = io.readTestData()
    
    print("Reading Holiday Database ...")
    holidayDF = io.readHoliday()
    
    print("Reading item list....")
    itemDF = io.readItemNbr()
    
    
    print("Read complete... ")
    finishTime = datetime.datetime.now().isoformat()
    print("Reading time ends:", finishTime)
    
else:  
    print("Skipping Read...")

if(dataWrangle):
    newItemDF = dW.pickItems(itemDF)
    
    currentTime = datetime.datetime.now().isoformat()
    print("Moving Averages time begin:", currentTime)
    
    print("Implementing moving averages...")
    #trainDF = dW.movingAverages(trainDF)
    #trainDF = dW.movingAverages(trainDF, newItemDF)
    newTrain = dW.newMovingAverages(trainDF, 90)
    
    finishTime = datetime.datetime.now().isoformat()
    print("Moving Averages time ends:", finishTime)

    newTrain['date'] = pd.to_datetime(newTrain['date'])
    newTrain = io.convertDate(newTrain)


    print("Adding Holidays...")
    trainDF = dW.addHolidays(newTrain, holidayDF)
  
    

#Creating Verification set
if(verify):
    currentTime = datetime.datetime.now().isoformat()
    print("Verification time begin:", currentTime)
    nTrainRows = trainDF.shape[0]
    
    train = trainDF.drop(['unit_sales_y'], axis = 1)
    train.columns = train.columns.str.replace("_x","")
    
    train = train.drop(['date'], axis=1)
    dummyTrain = pd.get_dummies(train)
    
    print("Set up verification tables...")
    train = dummyTrain[:int(nTrainRows*0.8)]
    verification = dummyTrain[int(nTrainRows*0.8):]
    
    #train.to_csv("../../data/Processed/0501-1028-trainSetProcessed.csv")
    #verification.to_csv("../../data/Processed/0501-1028-VerifyProcessed.csv")
    
    currentTime = datetime.datetime.now().isoformat()
    print("Verification time ends:", currentTime)
        
    
if(model == 0): #Do XGBoost
   
    print("Dropping date and item number...")

    #train = pd.read_csv("../../data/Processed/0402-1028-trainSetProcessed.csv")
    #verification = pd.read_csv("../../data/Processed/0402-1028-VerifyProcessed.csv")

    
    print("training XGBoost Model")
    XGBModel = xgbt.trainXGModel(train)
    
    #xgVerify = pd.read_csv("../../data/Processed/3012-1219-VerifyProcessed.csv")
    xgVerify = verification
    xgVerifyScores = xgVerify.unit_sales
    xgVerify = xgVerify.drop(['unit_sales'], axis=1)
    
    scores = xgVerifyScores.values #converts scores to values 
    
    #del xgTrain    
    
    predictions = XGBModel.predict(xgVerify)
    
    predictions[predictions < 0] = 0
    scores[scores < 0] = 0
    
    np.savetxt("../../predictions2.csv", predictions, delimiter = ",")
    
    loss = xgbt.loss(predictions, scores)
    
    