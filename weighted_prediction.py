# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 19:55:35 2019

@author: weihong
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 14:20:10 2019

@author: weihong
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from random import seed
from random import random
import matplotlib.pyplot as pyplot
from sklearn.metrics import roc_curve, auc, roc_auc_score

seed(9)

data = pd.read_csv("DataSheet.csv", encoding='latin1')
data['Date/Time'] = pd.to_datetime(data['Date/Time'])
training_data = data.loc[(data['Date/Time'] >= '2014-01-01') & (data['Date/Time'] <= '2015-12-31')]
test_data = data.loc[(data['Date/Time'] >= '2016-01-01')]
low_water_level_data = data.loc[(data['CategorizedWaterLevel'] == 'Low')  & (data['Date/Time'] >= '2014-01-01') & (data['Date/Time'] <= '2015-12-31')]
moderate_water_level_data = data.loc[(data['CategorizedWaterLevel'] == 'Moderate') & (data['Date/Time'] >= '2014-01-01') & (data['Date/Time'] <= '2015-12-31')]
moderately_high_water_level_data = data.loc[(data['CategorizedWaterLevel'] == 'Moderately High') & (data['Date/Time'] >= '2014-01-01') & (data['Date/Time'] <= '2015-12-31')]
high_water_level_data = data.loc[(data['CategorizedWaterLevel'] == 'High') & (data['Date/Time'] >= '2014-01-01') & (data['Date/Time'] <= '2015-12-31')]
very_high_water_level_data = data.loc[(data["CategorizedWaterLevel"] == "Very High")&(data["Date/Time"] >= "2014-01-01") & (data["Date/Time"] <= "2015-12-31")]

def predict_waterlevel(row):
    # rainfall = no rain
    if row['Categorized Rainfall '] == 'No Rain' and row['Categorized Temperature'] == 'Extreme Cold' and row['Categorized Wind'] == 'Low':
        return 'Low'
    if row['Categorized Rainfall '] == 'No Rain' and row['Categorized Temperature'] == 'Extreme Cold' and row['Categorized Wind'] == 'Moderate':
        return 'Low'
    if row['Categorized Rainfall '] == 'No Rain' and row['Categorized Temperature'] == 'Very Cold' and row['Categorized Wind'] == 'Low':
        return 'Low'
    if row['Categorized Rainfall '] == 'No Rain' and row['Categorized Temperature'] == 'Very Cold' and row['Categorized Wind'] == 'Moderate':
        return 'Low'
    if row['Categorized Rainfall '] == 'No Rain' and row['Categorized Temperature'] == 'Cold' and row['Categorized Wind'] == 'Low':
        return 'Low'
    if row['Categorized Rainfall '] == 'No Rain' and row['Categorized Temperature'] == 'Cold' and row['Categorized Wind'] == 'Moderate':
        return 'Low'
    
    if row['Categorized Rainfall '] == 'No Rain' and row['Categorized Temperature'] == 'Neutral' and row['Categorized Wind'] == 'Low':
        lowWeight = low_water_level_data.loc[(low_water_level_data['Categorized Rainfall '] == 'No Rain') & (low_water_level_data['Categorized Temperature'] == 'Neutral') & (low_water_level_data['Categorized Wind'] == 'Low')].shape[0] 
        modWeight =  moderate_water_level_data.loc[(moderate_water_level_data['Categorized Rainfall '] == 'No Rain') & (moderate_water_level_data['Categorized Temperature'] == 'Neutral') & (moderate_water_level_data['Categorized Wind'] == 'Low')].shape[0]
        modHighWeight =  moderately_high_water_level_data.loc[(moderately_high_water_level_data['Categorized Rainfall '] == 'No Rain') & (moderately_high_water_level_data['Categorized Temperature'] == 'Neutral') & (moderately_high_water_level_data['Categorized Wind'] == 'Low')].shape[0]
        vHighWeight = very_high_water_level_data.loc[(very_high_water_level_data['Categorized Rainfall '] == 'No Rain') & (very_high_water_level_data['Categorized Temperature'] == 'Neutral') & (very_high_water_level_data['Categorized Wind'] == 'Low')].shape[0]
        TotalWeight = lowWeight + modWeight + modHighWeight + vHighWeight
        randValue = random()
        if randValue < (lowWeight/TotalWeight):
            return 'Low'
        elif randValue >= (lowWeight/TotalWeight) and randValue < ((lowWeight+modWeight)/TotalWeight):
            return 'Moderate'
        elif randValue >= ((lowWeight+modWeight)/TotalWeight) and randValue < ((lowWeight+modWeight+modHighWeight)/TotalWeight):
            return 'Moderately High'
        elif randValue >= ((lowWeight+modWeight+modHighWeight)/TotalWeight) and randValue <= 1:
            return 'Very High'
        
    if row['Categorized Rainfall '] == 'No Rain' and row['Categorized Temperature'] == 'Neutral' and row['Categorized Wind'] == 'Moderate':
        lowWeight = low_water_level_data.loc[(low_water_level_data['Categorized Rainfall '] == 'No Rain') & (low_water_level_data['Categorized Temperature'] == 'Neutral') & (low_water_level_data['Categorized Wind'] == 'Moderate')].shape[0] 
        modWeight =  moderate_water_level_data.loc[(moderate_water_level_data['Categorized Rainfall '] == 'No Rain') & (moderate_water_level_data['Categorized Temperature'] == 'Neutral') & (moderate_water_level_data['Categorized Wind'] == 'Moderate')].shape[0]
        modHighWeight =  moderately_high_water_level_data.loc[(moderately_high_water_level_data['Categorized Rainfall '] == 'No Rain') & (moderately_high_water_level_data['Categorized Temperature'] == 'Neutral') & (moderately_high_water_level_data['Categorized Wind'] == 'Moderate')].shape[0]
        vHighWeight = very_high_water_level_data.loc[(very_high_water_level_data['Categorized Rainfall '] == 'No Rain') & (very_high_water_level_data['Categorized Temperature'] == 'Neutral') & (very_high_water_level_data['Categorized Wind'] == 'Moderate')].shape[0]
        TotalWeight = lowWeight + modWeight + modHighWeight + vHighWeight
        randValue = random()
        if randValue < (lowWeight/TotalWeight):
            return 'Low'
        elif randValue >= (lowWeight/TotalWeight) and randValue < ((lowWeight+modWeight)/TotalWeight):
            return 'Moderate'
        elif randValue >= ((lowWeight+modWeight)/TotalWeight) and randValue < ((lowWeight+modWeight+modHighWeight)/TotalWeight):
            return 'Moderately High'
        elif randValue >= ((lowWeight+modWeight+modHighWeight)/TotalWeight) and randValue <= 1:
            return 'Very High'
    
    if row['Categorized Rainfall '] == 'No Rain' and row['Categorized Temperature'] == 'Warm' and row['Categorized Wind'] == 'Low':
        modWeight =  moderate_water_level_data.loc[(moderate_water_level_data['Categorized Rainfall '] == 'No Rain') & (moderate_water_level_data['Categorized Temperature'] == 'Warm') & (moderate_water_level_data['Categorized Wind'] =='Low')].shape[0]
        modHighWeight =  moderately_high_water_level_data.loc[(moderately_high_water_level_data['Categorized Rainfall '] == 'No Rain') & (moderately_high_water_level_data['Categorized Temperature'] == 'Warm') & (moderately_high_water_level_data['Categorized Wind'] =='Low')].shape[0]
        highWeight = high_water_level_data.loc[(high_water_level_data['Categorized Rainfall '] == 'No Rain') & (high_water_level_data['Categorized Temperature'] == 'Warm') & (high_water_level_data['Categorized Wind'] =='Low')].shape[0]
        vHighWeight = very_high_water_level_data.loc[(very_high_water_level_data['Categorized Rainfall '] == 'No Rain') & (very_high_water_level_data['Categorized Temperature'] == 'Warm') & (very_high_water_level_data['Categorized Wind'] =='Low')].shape[0]
        TotalWeight = modWeight + modHighWeight + highWeight + vHighWeight
        randValue = random()
        if randValue < (modWeight/TotalWeight):
            return 'Moderate'
        elif randValue >= (modWeight/TotalWeight) and randValue < ((modWeight+modHighWeight)/TotalWeight):
            return 'Moderately High'
        elif randValue >= ((modWeight+modHighWeight)/TotalWeight) and randValue < ((modWeight+modHighWeight+highWeight)/TotalWeight):
            return 'High'
        elif randValue >= ((modWeight+modHighWeight+highWeight)/TotalWeight) and randValue <= 1:
            return 'Very High'
        
    if row['Categorized Rainfall '] == 'No Rain' and row['Categorized Temperature'] == 'Warm' and row['Categorized Wind'] == 'Moderate':
        modWeight =  moderate_water_level_data.loc[(moderate_water_level_data['Categorized Rainfall '] == 'No Rain') & (moderate_water_level_data['Categorized Temperature'] == 'Warm') & (moderate_water_level_data['Categorized Wind'] =='Moderate')].shape[0]
        modHighWeight =  moderately_high_water_level_data.loc[(moderately_high_water_level_data['Categorized Rainfall '] == 'No Rain') & (moderately_high_water_level_data['Categorized Temperature'] == 'Warm') & (moderately_high_water_level_data['Categorized Wind'] =='Moderate')].shape[0]
        highWeight = high_water_level_data.loc[(high_water_level_data['Categorized Rainfall '] == 'No Rain') & (high_water_level_data['Categorized Temperature'] == 'Warm') & (high_water_level_data['Categorized Wind'] =='Moderate')].shape[0]
        vHighWeight = very_high_water_level_data.loc[(very_high_water_level_data['Categorized Rainfall '] == 'No Rain') & (very_high_water_level_data['Categorized Temperature'] == 'Warm') & (very_high_water_level_data['Categorized Wind'] =='Moderate')].shape[0]
        TotalWeight = modWeight + modHighWeight + highWeight + vHighWeight
        randValue = random()
        if randValue < (modWeight/TotalWeight):
            return 'Moderate'
        elif randValue >= (modWeight/TotalWeight) and randValue < ((modWeight+modHighWeight)/TotalWeight):
            return 'Moderately High'
        elif randValue >= ((modWeight+modHighWeight)/TotalWeight) and randValue < ((modWeight+modHighWeight+highWeight)/TotalWeight):
            return 'High'
        elif randValue >= ((modWeight+modHighWeight+highWeight)/TotalWeight) and randValue <= 1:
            return 'Very High'
    
    if row['Categorized Rainfall '] == 'No Rain' and row['Categorized Temperature'] == 'Hot' and row['Categorized Wind'] == 'Low':
        modWeight =  moderate_water_level_data.loc[(moderate_water_level_data['Categorized Rainfall '] == 'No Rain') & (moderate_water_level_data['Categorized Temperature'] == 'Hot') & (moderate_water_level_data['Categorized Wind'] =='Low')].shape[0]
        modHighWeight =  moderately_high_water_level_data.loc[(moderately_high_water_level_data['Categorized Rainfall '] == 'No Rain') & (moderately_high_water_level_data['Categorized Temperature'] == 'Hot') & (moderately_high_water_level_data['Categorized Wind'] =='Low')].shape[0]
        highWeight = high_water_level_data.loc[(high_water_level_data['Categorized Rainfall '] == 'No Rain') & (high_water_level_data['Categorized Temperature'] == 'Hot') & (high_water_level_data['Categorized Wind'] =='Low')].shape[0]
        vHighWeight = very_high_water_level_data.loc[(very_high_water_level_data['Categorized Rainfall '] == 'No Rain') & (very_high_water_level_data['Categorized Temperature'] == 'Hot') & (very_high_water_level_data['Categorized Wind'] =='Low')].shape[0]
        TotalWeight = modWeight + modHighWeight + highWeight + vHighWeight
        randValue = random()
        if randValue < (modWeight/TotalWeight):
            return 'Moderate'
        elif randValue >= (modWeight/TotalWeight) and randValue < ((modWeight+modHighWeight)/TotalWeight):
            return 'Moderately High'
        elif randValue >= ((modWeight+modHighWeight)/TotalWeight) and randValue < ((modWeight+modHighWeight+highWeight)/TotalWeight):
            return 'High'
        elif randValue >= ((modWeight+modHighWeight+highWeight)/TotalWeight) and randValue <= 1:
            return 'Very High'
        
    if row['Categorized Rainfall '] == 'No Rain' and row['Categorized Temperature'] == 'Hot' and row['Categorized Wind'] == 'Moderate':
        modWeight =  moderate_water_level_data.loc[(moderate_water_level_data['Categorized Rainfall '] == 'No Rain') & (moderate_water_level_data['Categorized Temperature'] == 'Hot') & (moderate_water_level_data['Categorized Wind'] =='Moderate')].shape[0]
        modHighWeight =  moderately_high_water_level_data.loc[(moderately_high_water_level_data['Categorized Rainfall '] == 'No Rain') & (moderately_high_water_level_data['Categorized Temperature'] == 'Hot') & (moderately_high_water_level_data['Categorized Wind'] =='Moderate')].shape[0]
        highWeight = high_water_level_data.loc[(high_water_level_data['Categorized Rainfall '] == 'No Rain') & (high_water_level_data['Categorized Temperature'] == 'Hot') & (high_water_level_data['Categorized Wind'] =='Moderate')].shape[0]
        vHighWeight = very_high_water_level_data.loc[(very_high_water_level_data['Categorized Rainfall '] == 'No Rain') & (very_high_water_level_data['Categorized Temperature'] == 'Hot') & (very_high_water_level_data['Categorized Wind'] =='Moderate')].shape[0]
        TotalWeight = modWeight + modHighWeight + highWeight + vHighWeight
        randValue = random()
        if randValue < (modWeight/TotalWeight):
            return 'Moderate'
        elif randValue >= (modWeight/TotalWeight) and randValue < ((modWeight+modHighWeight)/TotalWeight):
            return 'Moderately High'
        elif randValue >= ((modWeight+modHighWeight)/TotalWeight) and randValue < ((modWeight+modHighWeight+highWeight)/TotalWeight):
            return 'High'
        elif randValue >= ((modWeight+modHighWeight+highWeight)/TotalWeight) and randValue <= 1:
            return 'Very High'
    
    #rainfall = drizzle
    if row['Categorized Rainfall '] == 'Drizzle' and row['Categorized Temperature'] == 'Extreme Cold' and row['Categorized Wind'] == 'Low':
        return 'Low'
    if row['Categorized Rainfall '] == 'Drizzle' and row['Categorized Temperature'] == 'Extreme Cold' and row['Categorized Wind'] == 'Moderate':
        return 'Low'
    if row['Categorized Rainfall '] == 'Drizzle' and row['Categorized Temperature'] == 'Very Cold' and row['Categorized Wind'] == 'Low':
        return 'Low'
    if row['Categorized Rainfall '] == 'Drizzle' and row['Categorized Temperature'] == 'Very Cold' and row['Categorized Wind'] == 'Moderate':
        return 'Low'
    if row['Categorized Rainfall '] == 'Drizzle' and row['Categorized Temperature'] == 'Cold' and row['Categorized Wind'] == 'Low':
        return 'Low'
    if row['Categorized Rainfall '] == 'Drizzle' and row['Categorized Temperature'] == 'Cold' and row['Categorized Wind'] == 'Moderate':
        return 'Low'
    
    if row['Categorized Rainfall '] == 'Drizzle' and row['Categorized Temperature'] == 'Neutral' and row['Categorized Wind'] == 'Low':
        lowWeight = low_water_level_data.loc[(low_water_level_data['Categorized Rainfall '] == 'Drizzle') & (low_water_level_data['Categorized Temperature'] == 'Neutral') & (low_water_level_data['Categorized Wind'] == 'Low')].shape[0] 
        modWeight =  moderate_water_level_data.loc[(moderate_water_level_data['Categorized Rainfall '] == 'Drizzle') & (moderate_water_level_data['Categorized Temperature'] == 'Neutral') & (moderate_water_level_data['Categorized Wind'] == 'Low')].shape[0]
        modHighWeight =  moderately_high_water_level_data.loc[(moderately_high_water_level_data['Categorized Rainfall '] == 'Drizzle') & (moderately_high_water_level_data['Categorized Temperature'] == 'Neutral') & (moderately_high_water_level_data['Categorized Wind'] == 'Low')].shape[0]
        vHighWeight = very_high_water_level_data.loc[(very_high_water_level_data['Categorized Rainfall '] == 'Drizzle') & (very_high_water_level_data['Categorized Temperature'] == 'Neutral') & (very_high_water_level_data['Categorized Wind'] == 'Low')].shape[0]
        TotalWeight = lowWeight + modWeight + modHighWeight + vHighWeight
        randValue = random()
        if randValue < (lowWeight/TotalWeight):
            return 'Low'
        elif randValue >= (lowWeight/TotalWeight) and randValue < ((lowWeight+modWeight)/TotalWeight):
            return 'Moderate'
        elif randValue >= ((lowWeight+modWeight)/TotalWeight) and randValue < ((lowWeight+modWeight+modHighWeight)/TotalWeight):
            return 'Moderately High'
        elif randValue >= ((lowWeight+modWeight+modHighWeight)/TotalWeight) and randValue <= 1:
            return 'Very High'
        
    if row['Categorized Rainfall '] == 'Drizzle' and row['Categorized Temperature'] == 'Neutral' and row['Categorized Wind'] == 'Moderate':
        lowWeight = low_water_level_data.loc[(low_water_level_data['Categorized Rainfall '] == 'Drizzle') & (low_water_level_data['Categorized Temperature'] == 'Neutral') & (low_water_level_data['Categorized Wind'] == 'Moderate')].shape[0] 
        modWeight =  moderate_water_level_data.loc[(moderate_water_level_data['Categorized Rainfall '] == 'Drizzle') & (moderate_water_level_data['Categorized Temperature'] == 'Neutral') & (moderate_water_level_data['Categorized Wind'] == 'Moderate')].shape[0]
        modHighWeight =  moderately_high_water_level_data.loc[(moderately_high_water_level_data['Categorized Rainfall '] == 'Drizzle') & (moderately_high_water_level_data['Categorized Temperature'] == 'Neutral') & (moderately_high_water_level_data['Categorized Wind'] == 'Moderate')].shape[0]
        vHighWeight = very_high_water_level_data.loc[(very_high_water_level_data['Categorized Rainfall '] == 'Drizzle') & (very_high_water_level_data['Categorized Temperature'] == 'Neutral') & (very_high_water_level_data['Categorized Wind'] == 'Moderate')].shape[0]
        TotalWeight = lowWeight + modWeight + modHighWeight + vHighWeight
        randValue = random()
        if randValue < (lowWeight/TotalWeight):
            return 'Low'
        elif randValue >= (lowWeight/TotalWeight) and randValue < ((lowWeight+modWeight)/TotalWeight):
            return 'Moderate'
        elif randValue >= ((lowWeight+modWeight)/TotalWeight) and randValue < ((lowWeight+modWeight+modHighWeight)/TotalWeight):
            return 'Moderately High'
        elif randValue >= ((lowWeight+modWeight+modHighWeight)/TotalWeight) and randValue <= 1:
            return 'Very High'
    
    if row['Categorized Rainfall '] == 'Drizzle' and row['Categorized Temperature'] == 'Warm' and row['Categorized Wind'] == 'Low':
        modWeight =  moderate_water_level_data.loc[(moderate_water_level_data['Categorized Rainfall '] == 'Drizzle') & (moderate_water_level_data['Categorized Temperature'] == 'Warm') & (moderate_water_level_data['Categorized Wind'] =='Low')].shape[0]
        modHighWeight =  moderately_high_water_level_data.loc[(moderately_high_water_level_data['Categorized Rainfall '] == 'Drizzle') & (moderately_high_water_level_data['Categorized Temperature'] == 'Warm') & (moderately_high_water_level_data['Categorized Wind'] =='Low')].shape[0]
        highWeight = high_water_level_data.loc[(high_water_level_data['Categorized Rainfall '] == 'Drizzle') & (high_water_level_data['Categorized Temperature'] == 'Warm') & (high_water_level_data['Categorized Wind'] =='Low')].shape[0]
        vHighWeight = very_high_water_level_data.loc[(very_high_water_level_data['Categorized Rainfall '] == 'Drizzle') & (very_high_water_level_data['Categorized Temperature'] == 'Warm') & (very_high_water_level_data['Categorized Wind'] =='Low')].shape[0]
        TotalWeight = modWeight + modHighWeight + highWeight + vHighWeight
        randValue = random()
        if randValue < (modWeight/TotalWeight):
            return 'Moderate'
        elif randValue >= (modWeight/TotalWeight) and randValue < ((modWeight+modHighWeight)/TotalWeight):
            return 'Moderately High'
        elif randValue >= ((modWeight+modHighWeight)/TotalWeight) and randValue < ((modWeight+modHighWeight+highWeight)/TotalWeight):
            return 'High'
        elif randValue >= ((modWeight+modHighWeight+highWeight)/TotalWeight) and randValue <= 1:
            return 'Very High'
        
    if row['Categorized Rainfall '] == 'Drizzle' and row['Categorized Temperature'] == 'Warm' and row['Categorized Wind'] == 'Moderate':
        modWeight =  moderate_water_level_data.loc[(moderate_water_level_data['Categorized Rainfall '] == 'Drizzle') & (moderate_water_level_data['Categorized Temperature'] == 'Warm') & (moderate_water_level_data['Categorized Wind'] =='Moderate')].shape[0]
        modHighWeight =  moderately_high_water_level_data.loc[(moderately_high_water_level_data['Categorized Rainfall '] == 'Drizzle') & (moderately_high_water_level_data['Categorized Temperature'] == 'Warm') & (moderately_high_water_level_data['Categorized Wind'] =='Moderate')].shape[0]
        highWeight = high_water_level_data.loc[(high_water_level_data['Categorized Rainfall '] == 'Drizzle') & (high_water_level_data['Categorized Temperature'] == 'Warm') & (high_water_level_data['Categorized Wind'] =='Moderate')].shape[0]
        vHighWeight = very_high_water_level_data.loc[(very_high_water_level_data['Categorized Rainfall '] == 'Drizzle') & (very_high_water_level_data['Categorized Temperature'] == 'Warm') & (very_high_water_level_data['Categorized Wind'] =='Moderate')].shape[0]
        TotalWeight = modWeight + modHighWeight + highWeight + vHighWeight
        randValue = random()
        if randValue < (modWeight/TotalWeight):
            return 'Moderate'
        elif randValue >= (modWeight/TotalWeight) and randValue < ((modWeight+modHighWeight)/TotalWeight):
            return 'Moderately High'
        elif randValue >= ((modWeight+modHighWeight)/TotalWeight) and randValue < ((modWeight+modHighWeight+highWeight)/TotalWeight):
            return 'High'
        elif randValue >= ((modWeight+modHighWeight+highWeight)/TotalWeight) and randValue <= 1:
            return 'Very High'
    
    if row['Categorized Rainfall '] == 'Drizzle' and row['Categorized Temperature'] == 'Hot' and row['Categorized Wind'] == 'Low':
        modWeight =  moderate_water_level_data.loc[(moderate_water_level_data['Categorized Rainfall '] == 'Drizzle') & (moderate_water_level_data['Categorized Temperature'] == 'Hot') & (moderate_water_level_data['Categorized Wind'] =='Low')].shape[0]
        modHighWeight =  moderately_high_water_level_data.loc[(moderately_high_water_level_data['Categorized Rainfall '] == 'Drizzle') & (moderately_high_water_level_data['Categorized Temperature'] == 'Hot') & (moderately_high_water_level_data['Categorized Wind'] =='Low')].shape[0]
        highWeight = high_water_level_data.loc[(high_water_level_data['Categorized Rainfall '] == 'Drizzle') & (high_water_level_data['Categorized Temperature'] == 'Hot') & (high_water_level_data['Categorized Wind'] =='Low')].shape[0]
        vHighWeight = very_high_water_level_data.loc[(very_high_water_level_data['Categorized Rainfall '] == 'Drizzle') & (very_high_water_level_data['Categorized Temperature'] == 'Hot') & (very_high_water_level_data['Categorized Wind'] =='Low')].shape[0]
        TotalWeight = modWeight + modHighWeight + highWeight + vHighWeight
        randValue = random()
        if randValue < (modWeight/TotalWeight):
            return 'Moderate'
        elif randValue >= (modWeight/TotalWeight) and randValue < ((modWeight+modHighWeight)/TotalWeight):
            return 'Moderately High'
        elif randValue >= ((modWeight+modHighWeight)/TotalWeight) and randValue < ((modWeight+modHighWeight+highWeight)/TotalWeight):
            return 'High'
        elif randValue >= ((modWeight+modHighWeight+highWeight)/TotalWeight) and randValue <= 1:
            return 'Very High'
        
    if row['Categorized Rainfall '] == 'Drizzle' and row['Categorized Temperature'] == 'Hot' and row['Categorized Wind'] == 'Moderate':
        modWeight =  moderate_water_level_data.loc[(moderate_water_level_data['Categorized Rainfall '] == 'Drizzle') & (moderate_water_level_data['Categorized Temperature'] == 'Hot') & (moderate_water_level_data['Categorized Wind'] =='Moderate')].shape[0]
        modHighWeight =  moderately_high_water_level_data.loc[(moderately_high_water_level_data['Categorized Rainfall '] == 'Drizzle') & (moderately_high_water_level_data['Categorized Temperature'] == 'Hot') & (moderately_high_water_level_data['Categorized Wind'] =='Moderate')].shape[0]
        highWeight = high_water_level_data.loc[(high_water_level_data['Categorized Rainfall '] == 'Drizzle') & (high_water_level_data['Categorized Temperature'] == 'Hot') & (high_water_level_data['Categorized Wind'] =='Moderate')].shape[0]
        vHighWeight = very_high_water_level_data.loc[(very_high_water_level_data['Categorized Rainfall '] == 'Drizzle') & (very_high_water_level_data['Categorized Temperature'] == 'Hot') & (very_high_water_level_data['Categorized Wind'] =='Moderate')].shape[0]
        TotalWeight = modWeight + modHighWeight + highWeight + vHighWeight
        randValue = random()
        if randValue < (modWeight/TotalWeight):
            return 'Moderate'
        elif randValue >= (modWeight/TotalWeight) and randValue < ((modWeight+modHighWeight)/TotalWeight):
            return 'Moderately High'
        elif randValue >= ((modWeight+modHighWeight)/TotalWeight) and randValue < ((modWeight+modHighWeight+highWeight)/TotalWeight):
            return 'High'
        elif randValue >= ((modWeight+modHighWeight+highWeight)/TotalWeight) and randValue <= 1:
            return 'Very High'
    
test_data['PredictedWaterLevel'] = test_data.apply (lambda row: predict_waterlevel(row), axis=1)
test_data['PredictedWaterLevel'] = test_data['PredictedWaterLevel'].fillna('Low')

WATER_LEVEL_CATS = ["Low", "Moderate", "Moderately High", "High", "Very High"]

WATER_LEVEL_ENUM = {
    "Low": 0, 
    "Moderate": 1, 
    "Moderately High": 2, 
    "High": 3, 
    "Very High": 4
}

prediction_data = {}
for water_level in WATER_LEVEL_CATS:
    prediction_data[water_level] = {}
    prediction_data[water_level]["correct_pred"] = test_data.loc[(test_data["CategorizedWaterLevel"] == water_level) & (test_data["CategorizedWaterLevel"] == test_data["PredictedWaterLevel"])].shape[0]  #Count of correct predictions for this water level class
    prediction_data[water_level]["total_pred"] = test_data.loc[(test_data["PredictedWaterLevel"] == water_level)].shape[0] #Count of total predictions for this water level class
    prediction_data[water_level]["actual_cnt"] = test_data.loc[(test_data["CategorizedWaterLevel"] == water_level)].shape[0] #Count of actual instances of this water level class

predicted_classes = np.zeros((test_data.shape[0], len(WATER_LEVEL_CATS)), dtype=np.dtype(int))
actual_classes = np.zeros((test_data.shape[0], len(WATER_LEVEL_CATS)), dtype=np.dtype(int))

row_num = 0
for index, row in test_data.iterrows():
    actual_water_level = row["CategorizedWaterLevel"]
    predicted_water_level = row["PredictedWaterLevel"]

    #Set "bit" for predicted class in this entries "predicted value" vector to 1
    predicted_classes[row_num][WATER_LEVEL_ENUM[predicted_water_level]] = 1
    
    #Set "bit" for actual class in this entries "actual value" vector to 1
    actual_classes[row_num][WATER_LEVEL_ENUM[actual_water_level]] = 1
    
    #Increment row number counter
    row_num = row_num + 1
    
    
for water_level in WATER_LEVEL_CATS:
    wl_pred_data = prediction_data[water_level]
    #If-else statements provide safeguards for "divide-by-zero" cases
    if wl_pred_data["total_pred"] == 0:
        wl_pred_data["precision"] = 0.0
    else:
        wl_pred_data["precision"] = wl_pred_data["correct_pred"]/wl_pred_data["total_pred"]
        
    if wl_pred_data["actual_cnt"] == 0:
        wl_pred_data["recall"] = 0.0
    else:
        wl_pred_data["recall"] = wl_pred_data["correct_pred"]/wl_pred_data["actual_cnt"]
        
    if wl_pred_data["precision"] + wl_pred_data["recall"] == 0:
        wl_pred_data["f1_score"] = 0.0
    else:
        wl_pred_data["f1_score"] = 2 * ((wl_pred_data["precision"] * wl_pred_data["recall"])/(wl_pred_data["precision"] + wl_pred_data["recall"]))
        
#Calculate macro-precision, macro-recall, and macro-F1 score for prediction model (macro = macro-average)
macro_precision_score_sum = 0.0
macro_recall_score_sum = 0.0
macro_f1_score_sum = 0.0
for water_level in WATER_LEVEL_CATS:
    wl_pred_data = prediction_data[water_level]
    macro_precision_score_sum = macro_precision_score_sum + wl_pred_data["precision"]
    macro_recall_score_sum = macro_recall_score_sum + wl_pred_data["recall"]
    macro_f1_score_sum = macro_f1_score_sum + wl_pred_data["f1_score"]
macro_precision_score = macro_precision_score_sum/len(WATER_LEVEL_CATS)
macro_recall_score = macro_recall_score_sum/len(WATER_LEVEL_CATS)
macro_f1_score = macro_f1_score_sum/len(WATER_LEVEL_CATS)

#Following ROC-AUC computation code is based on example code found at the following URL:
#https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

#Compute ROC curve & AUC (area under curve) value for each water level class
fpr = {} #False positive rates
tpr = {} #True positive rates
roc_auc = {} #AUC of ROC curves
for index, water_level in enumerate(WATER_LEVEL_CATS):
    fpr[water_level], tpr[water_level], _ = roc_curve(actual_classes[:, index], predicted_classes[:, index])
    roc_auc[water_level] = auc(fpr[water_level], tpr[water_level])
    
#Compute macro-average ROC curve and AUC values for prediction model
all_fpr = np.unique(np.concatenate([fpr[wl] for wl in WATER_LEVEL_CATS]))
mean_tpr = np.zeros_like(all_fpr)
for wl in WATER_LEVEL_CATS:
    mean_tpr += np.interp(all_fpr, fpr[wl], tpr[wl])
mean_tpr /= len(WATER_LEVEL_CATS)
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

#Print prediction results (accuracy of model)
print("Prediction Model Results:")
print("=========================")
print("Macro-Precision Score:", macro_precision_score)
print("Macro-Recall Score:", macro_recall_score)
print("Macro-F1 Score:", macro_f1_score)
print("Macro-ROC-AUC Score:", roc_auc["macro"], "\n")

#Print more detailed prediction results
print("Individual Class Results:")
print("=========================")
for water_level in WATER_LEVEL_CATS:
    wl_pred_data = prediction_data[water_level]
    print("  " + water_level + ":")
    print("    Precision Score:", wl_pred_data["precision"])
    print("    Recall Score:", wl_pred_data["recall"])
    print("    F1 Score:", wl_pred_data["f1_score"])
    print("    ROC-AUC Score:", roc_auc[water_level])
    
#Create graph plots of ROC curves (per-class & macro-average) and show to user
GRAPH_COLOURS = ["orange", "blue", "green", "cyan", "magenta"]
pyplot.figure()
pyplot.plot(fpr["macro"], tpr["macro"],
            label = 'Macro-average ROC curve (AUC = {0:0.4f})'.format(roc_auc["macro"]),
            color='red', linestyle=':', linewidth=2)
for index, water_level in enumerate(WATER_LEVEL_CATS):
    pyplot.plot(fpr[water_level], tpr[water_level],
            label = 'ROC curve of class {0} (AUC = {1:0.4f})'.format(water_level, roc_auc[water_level]),
            color=GRAPH_COLOURS[index], linestyle='-', linewidth=2)
pyplot.plot([0, 1], [0, 1], 'k--')
pyplot.xlim([0.0, 1.0])
pyplot.ylim([0.0, 1.0])
pyplot.xlabel("False Positive Rate")
pyplot.ylabel("True Positive Rate")
pyplot.title("Water Level Prediction Model ROC Curves")
pyplot.legend(loc="lower right")
pyplot.show()
