# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 13:46:57 2019

@author: weihong
"""

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from sklearn.metrics import roc_curve, auc, roc_auc_score

print("Loading data...\n")

data = pd.read_csv("Python_Data/DataSheet.csv", encoding='latin1')
data['Date/Time'] = pd.to_datetime(data['Date/Time'])
#2014-2015 records used for training prediction model, 2016 data is used for testing prediction model
training_data = data.loc[(data["Date/Time"] >= "2014-01-01") & (data["Date/Time"] <= "2015-12-31")]
test_data = data.loc[(data["Date/Time"] >= "2016-01-01") & (data["Date/Time"] <= "2016-12-31")]

#Constant category arrays
WATER_LEVEL_CATS = ["Low", "Moderate", "Moderately High", "High", "Very High"]
TEMP_LEVEL_CATS = ["Extreme Cold", "Very Cold", "Cold", "Neutral", "Warm", "Hot", "Very Hot"]
WIND_LEVEL_CATS = ["Low", "Moderate", "Moderately High", "High", "Extreme"]
RAIN_LEVEL_CATS = ["No Rain", "Drizzle", "Light Rain", "Moderate Rain", "Heavy Rain", "Violent Rain"]

#"Enum" for water level classes
WATER_LEVEL_ENUM = {
    "Low": 0, 
    "Moderate": 1, 
    "Moderately High": 2, 
    "High": 3, 
    "Very High": 4
}

#========== TRAINING ==========

print("Training prediction model...\n")

#Calculate weights off training data
weights = {}

for water_level in WATER_LEVEL_CATS:
    weights[water_level] = {"temp":{}, "wind":{}, "rain":{}}
    curr_wl_data = training_data.loc[(training_data["CategorizedWaterLevel"] == water_level)] #All data entries with current water level
    total_wl_occurence = curr_wl_data.shape[0] #Total count of entries with current water level
    
    for temp_level in TEMP_LEVEL_CATS:
        count_tl_and_wl = curr_wl_data.loc[(curr_wl_data["Categorized Temperature"] == temp_level)].shape[0] #Count of entries with current water level AND temp level
        weights[water_level]["temp"][temp_level] = count_tl_and_wl/total_wl_occurence #Weight calculation for current temp level for current water level
        
    for wind_level in WIND_LEVEL_CATS:
        count_wl_and_wl = curr_wl_data.loc[(curr_wl_data["Categorized Wind"] == wind_level)].shape[0] #Count of entries with current water level AND wind level
        weights[water_level]["wind"][wind_level] = count_wl_and_wl/total_wl_occurence #Weight calculation for current wind level for current water level
        
    for rain_level in RAIN_LEVEL_CATS:
        count_rl_and_wl = curr_wl_data.loc[(curr_wl_data["Categorized Rainfall "] == rain_level)].shape[0] #Count of entries with current water level AND rain level
        weights[water_level]["rain"][rain_level] = count_rl_and_wl/total_wl_occurence #Weight calculation for current rain level for current water level

#========== TESTING ==========
        
print("Testing prediction model...\n")

#Setup prediction data to use later for calculating model accuracy metrics
prediction_data = {}
for water_level in WATER_LEVEL_CATS:
    prediction_data[water_level] = {}
    prediction_data[water_level]["correct_pred"] = 0 #Count of correct predictions for this water level class
    prediction_data[water_level]["total_pred"] = 0 #Count of total predictions for this water level class
    prediction_data[water_level]["actual_cnt"] = test_data.loc[(test_data["CategorizedWaterLevel"] == water_level)].shape[0] #Count of actual instances of this water level class
    
#Track class predictions vs. actual class using vectors for computing ROC & AUC scores later
predicted_classes = np.zeros((test_data.shape[0], len(WATER_LEVEL_CATS)), dtype=np.dtype(int))
actual_classes = np.zeros((test_data.shape[0], len(WATER_LEVEL_CATS)), dtype=np.dtype(int))

#Use the sum of weights approach to predict water level of 2016 test data
row_num = 0
for index, row in test_data.iterrows():
    actual_water_level = row["CategorizedWaterLevel"]
    
    max_weight_sum = -1
    max_weight_wl = None #Prediction value is stored here
    for water_level in WATER_LEVEL_CATS:
        #Calculate the sum of weights using attribute categories for this row
        curr_weight_sum = weights[water_level]["temp"][row["Categorized Temperature"]] + \
                            weights[water_level]["wind"][row["Categorized Wind"]] + \
                            weights[water_level]["rain"][row["Categorized Rainfall "]]
        #Compare current weight sum to max weight sum, keep track of water level with greatest sum
        if curr_weight_sum > max_weight_sum:
            max_weight_wl = water_level
            max_weight_sum = curr_weight_sum
            
    #Increment total # of predictions for predicted water level class
    prediction_data[max_weight_wl]["total_pred"] = prediction_data[max_weight_wl]["total_pred"] + 1
    
    #Compare predicted water level to actual water level & count correct predictions
    if max_weight_wl == actual_water_level:
        prediction_data[max_weight_wl]["correct_pred"] = prediction_data[max_weight_wl]["correct_pred"] + 1
        
    #Set "bit" for predicted class in this entries "predicted value" vector to 1
    predicted_classes[row_num][WATER_LEVEL_ENUM[max_weight_wl]] = 1
    
    #Set "bit" for actual class in this entries "actual value" vector to 1
    actual_classes[row_num][WATER_LEVEL_ENUM[actual_water_level]] = 1
    
    #Increment row number counter
    row_num = row_num + 1
    
#========== RESULTS ==========
    
#Calculate precision, recall, and F1 score values for each water level class
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


