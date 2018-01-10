import pandas as pd
import datetime as dt

'''
readTrainData reads adjusted training Data After being adjusted by SQL joins 

With Mode 0, it reads the entire training data
With Mode 1, it reads training data that lasts a year
With Mode 2, it reads training data that lasts 3 months (for testing purposes)
'''
def readTrainData(mode):
    if (mode == 0):
        trainCSV = pd.read_csv("../../data/Processed/fullItemsTrain.csv")
    elif (mode == 1):
        trainCSV = pd.read_csv("../../data/Processed/082016itemsTrain.csv")
    elif (mode == 2):
        trainCSV = pd.read_csv("../../data/Processed/smallTrain.csv")
        
    trainCSV['date'] = pd.to_datetime(trainCSV['date'])
    trainCSV = convertDate(trainCSV)

    return trainCSV
'''
readHoliday reads data from the holiday table
'''
def readHoliday():
    holidayCSV = pd.read_csv("../../data/Original/holidays_events.csv")
    
    holidayCSV['date'] = pd.to_datetime(holidayCSV['date'])
    holidayCSV = convertDate(holidayCSV)
    
    return holidayCSV

def readTestData():
    testCSV = pd.read_csv("../../data/Processed/test.csv")
    
    testCSV['date'] = pd.to_datetime(testCSV['date'])
    testCSV = convertDate(testCSV)
    
    return testCSV

def convertDate(data):
    data['year'] = data['date'].apply(getYear)
    data['month'] = data['date'].apply(getMonth)
    data['day'] = data['date'].apply(getDay)
    
    return data
    

def getYear(x):
    return x.year

def getMonth(x):
    return x.month

def getDay(x):
    return x.day
