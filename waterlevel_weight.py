# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import csv
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt

#Getting Vertical weight
def getVWeight(weights, totalData, cat_name, categories):    
    print ('\n'+cat_name)
    for cat in categories:
        weights[cat] = {"Low":{}, "Moderate":{}, "Moderately High":{}, "High":{}, "Very High":{}}
        cat_data = totalData.loc[totalData[cat_name] == cat]
        print(cat,' ',cat_data.shape[0])
        if (cat_data.shape[0] > 0):
            low = cat_data.loc[cat_data["CategorizedWaterLevel"] == 'Low']
            mod = cat_data.loc[cat_data["CategorizedWaterLevel"] == 'Moderate']
            mod_high = cat_data.loc[cat_data["CategorizedWaterLevel"] == 'Moderately High']
            high = cat_data.loc[cat_data["CategorizedWaterLevel"] == 'High']
            very_high = cat_data.loc[cat_data["CategorizedWaterLevel"] == 'Very High']
            weights[cat]['Low'] = low.shape[0]/cat_data.shape[0]
            weights[cat]['Moderate'] = mod.shape[0]/cat_data.shape[0]
            weights[cat]['Moderately High'] = mod_high.shape[0]/cat_data.shape[0]
            weights[cat]['High'] = high.shape[0]/cat_data.shape[0]
            weights[cat]['Very High'] = very_high.shape[0]/cat_data.shape[0]
            print ('[', low.shape[0], mod.shape[0], mod_high.shape[0], high.shape[0], very_high.shape[0], ']')
            print ('[%.6f %.6f %.6f %.6f %.6f ]'% (low.shape[0]/cat_data.shape[0], mod.shape[0]/cat_data.shape[0], mod_high.shape[0]/cat_data.shape[0], \
                     high.shape[0]/cat_data.shape[0], very_high.shape[0]/cat_data.shape[0]))
        else:
            weights[cat]['Low'] = 0
            weights[cat]['Moderate'] = 0
            weights[cat]['Moderately High'] = 0
            weights[cat]['High'] = 0
            weights[cat]['Very High'] = 0
            print('[ 0 0 0 0 0]')
        
def rocGraph(y_true, y_score):
    auc = metrics.roc_auc_score(y_true, y_score)
    print("AUC: " + auc)
    plt.title("Receiver Operating Characteristic")
    fpr, tpr, threshold = metrics.roc_curve(y_true, y_score)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.plot(fpr, tpr)
    plt.show()


data = pd.read_csv("finalCategorizedData.csv", encoding='latin1')

training_model = data.loc[(data['Date/Time'] >= '2014-01-01') & (data['Date/Time'] <= '2015-12-31')]
test_model = data.loc[(data['Date/Time'] >= '2016-01-01') & (data['Date/Time'] <= '2016-12-31')]

weights = {}

rain_cat = ['No Rain','Drizzle','Light Rain','Moderate Rain','Heavy Rain','Violent Rain']
temp_cat = ['Extreme Cold','Very Cold','Cold','Neutral','Warm','Hot', 'Very Hot']
wind_cat = ['Low','Moderate','Moderately High','High','Extreme']
water_cat = ['Low','Moderate','Moderately High','High','Very High']

getVWeight(weights, training_model, 'Categorized Rainfall ', rain_cat)
getVWeight(weights, training_model, 'Categorized Temperature', temp_cat)
getVWeight(weights, training_model, 'Categorized Wind', wind_cat)

#========== TESTING ==========

print("Testing prediction model...\n")

#Use the sum of weights to predict water level of 2016 test data
total_correct_predictions = 0
wl_correct = {}
wl_total = {}
wl_actual = {}
wl_prec = {}
wl_recall = {}
wl_f1 = {}

predicted_classes = np.zeros((test_model.shape[0], len(water_cat)), dtype=np.dtype(int))
actual_classes = np.zeros((test_model.shape[0], len(water_cat)), dtype=np.dtype(int))
WATER_LEVEL_ENUM = {"Low": 0, "Moderate": 1, "Moderately High": 2, "High": 3, "Very High": 4}
row_num = 0

for water_level in water_cat:
    wl_correct[water_level] = 0
    wl_total[water_level] = 0
    wl_actual[water_level] = test_model.loc[(test_model["CategorizedWaterLevel"] == water_level)].shape[0]
    wl_recall[water_level] = 0
    #wl_f1[water_level] = 0

for index, row in test_model.iterrows():
    actual_water_level = row["CategorizedWaterLevel"]

    max_weight_sum = -1
    max_weight_wl = None
    for water_level in water_cat:    
        curr_weight_sum = weights[row["Categorized Rainfall "]][water_level] + weights[row["Categorized Temperature"]][water_level] \
                        + weights[row["Categorized Wind"]][water_level]
        if curr_weight_sum > max_weight_sum:
            max_weight_wl = water_level
            max_weight_sum = curr_weight_sum
    
    wl_total[max_weight_wl] += 1
    if max_weight_wl == actual_water_level:
        total_correct_predictions = total_correct_predictions + 1
        if max_weight_wl == 'Low':
            wl_correct[max_weight_wl] += 1
        elif max_weight_wl == 'Moderate':
            wl_correct[max_weight_wl] += 1
        elif max_weight_wl == 'Moderately High':
            wl_correct[max_weight_wl] += 1
        elif max_weight_wl == 'High':
            wl_correct[max_weight_wl] += 1
        else:
            wl_correct[max_weight_wl] += 1
            
    predicted_classes[row_num][WATER_LEVEL_ENUM[max_weight_wl]] = 1
    actual_classes[row_num][WATER_LEVEL_ENUM[actual_water_level]] = 1

    row_num += 1
    
#result   
for water_level in water_cat:
    #precision
    if wl_total[water_level] == 0:
        wl_prec[water_level] = 0;
    else:
        wl_prec[water_level] = wl_correct[water_level]/wl_total[water_level]
    
    #recall
    if wl_actual[water_level] == 0:
        wl_recall[water_level] = 0;
    else:
        wl_recall[water_level] = wl_correct[water_level]/wl_actual[water_level]
    
    #f1
    if wl_prec[water_level] + wl_recall[water_level] == 0:
        wl_f1[water_level] = 0;
    else:
        wl_f1[water_level] = (2 * wl_prec[water_level] * wl_recall[water_level]) / (wl_prec[water_level] + wl_recall[water_level])
    
fpr = {} #False positive rates
tpr = {} #True positive rates
roc_auc = {} #AUC of ROC curves
col = ['y--','c--','b--','r--','m--']
for index, water_level in enumerate(water_cat):
    fpr[water_level], tpr[water_level], threshold = metrics.roc_curve(actual_classes[:, index], predicted_classes[:, index])
    plt.plot(fpr[water_level], tpr[water_level], col[index], label=water_level)
    roc_auc[water_level] = metrics.auc(fpr[water_level], tpr[water_level])
    
all_fpr = np.unique(np.concatenate([fpr[wl] for wl in water_cat]))
mean_tpr = np.zeros_like(all_fpr)

for wl in water_cat:
    mean_tpr += np.interp(all_fpr, fpr[wl], tpr[wl])
    
mean_tpr /= len(water_cat)
fpr["total"] = all_fpr
tpr["total"] = mean_tpr
roc_auc["total"] = metrics.auc(fpr["total"], tpr["total"])
plt.plot(fpr["total"], tpr["total"], 'k-', label='Total')
    
print(wl_correct)
print("Prediction results:")
print("===================")
print("Total # of correct predictions:", total_correct_predictions)
print("Total # of test data entries:", test_model.shape[0])
      
print("\nIndividual prediction result")
for water_level in water_cat:
    print("\nWater Level: ",water_level)
    print("Precision: ",(wl_prec[water_level]))
    print("Recall: ",wl_recall[water_level])
    print("F1: ",wl_f1[water_level])
    print("AUC: ", roc_auc[water_level])
    
print("\nTotal prediction result")
t_prec = 0
t_recall = 0
t_f1 = 0
for water_level in water_cat:
    t_prec += wl_prec[water_level]
    t_recall += wl_recall[water_level]
    t_f1 += wl_f1[water_level]
print("Total Precision: ",t_prec/5)
print("Total Recall: ",t_recall/5)
print("Total F1: ",t_f1/5)
print("AUC: ", roc_auc["total"])

plt.title("Receiver Operating Characteristic")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc="lower right")
plt.show()