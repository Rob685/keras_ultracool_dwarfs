#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 17:29:00 2018

@author: Helios
"""
#see console2a
import pandas as pd
import numpy as np
from collections import Counter
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from itertools import combinations
import seaborn as sns
import warnings
warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
from pylab import rcParams
rcParams['figure.figsize'] = 10,5

train = pd.read_csv('/Users/Helios/Desktop/Research/CoolStars_Data/XGBoost_Classifier_TSET.csv', index_col=0)#Training labeled set
list(train)
train = train.replace('Giant','Contaminant')

uobjs = pd.read_csv('/Users/Helios/Desktop/Research/CoolStars_Data/unmatched_objs.csv', index_col=0)#Gaia unmached objs
list(uobjs)

#Target data (SkyMapper):
skymapper = pd.read_csv('/Users/Helios/Desktop/Research/CoolStars_Data/xmatch_colors_wuncert_new.csv', index_col=0)
list(skymapper)

##################################################################

#Here x is the training set (labeled) and y is the test set (any data set with appropriate column names):

def XGBooster(x,y):
    #Cleaning:
    tmags = x[['i_psf', 'z_psf',
               'h_m', 'j_m', 'k_m', 
               'w1mpro','w2mpro', 'StarType']] #Dataset NEEDS to have these column names for this function to work.
    tmags2 = tmags.dropna(how='any')
    tmags3 = tmags2[['i_psf', 'z_psf', 
                     'h_m', 'j_m', 'k_m', 
                     'w1mpro','w2mpro']]
    
    test_mags = y[['i_psf', 'z_psf', 
                   'h_m', 'j_m', 'k_m', 
                   'w1mpro', 'w2mpro']]#all skymapper
    test_mags2 = test_mags.dropna(how='any')
    
    #Combination function:
    def ccombinator(y):
        mags1 = pd.DataFrame(index=y.index)
        mags2 = pd.DataFrame(index=y.index)
        for a, b, in combinations(y.columns, 2):
            mags1['{}-{}'.format(a, b)] = y[a] - y[b]
            mags2['{}-{}'.format(b, a)] = y[b] - y[a]
        c = mags1.join(mags2)
        return c
    
    train_colors = ccombinator(tmags3).join(tmags3,how='outer')
    test_colors = ccombinator(test_mags2).join(test_mags2,how='outer')
    
    ytarget = tmags2.StarType
    
    test_size = 0.50
    X_train, X_test, y_train, y_test = train_test_split(train_colors, ytarget, test_size=test_size)
        
    #fit model no training data
        
    model = XGBClassifier()
    model.fit(X_train, y_train)
        
    #make predictions for test data
    predictions1 = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions1)
        
    #Predictions: train predictions (training set)
    conmatrix = confusion_matrix(y_test,predictions1)
    
    #Test predictions (SkyMapper)
    test_predictions = model.predict(test_colors)
    
    #F_Score Plot:
    plot_importance(model)
    plt.show()
    
    #Confusion Matrix Plot:
    conmatrix_df = pd.DataFrame.from_items([('Contaminants', conmatrix[0]), ('Dwarfs', conmatrix[1])], columns=['Pred. Contaminants', 'Pred. Dwarfs'], orient='index')
    print(conmatrix_df)
    
    #Confusion Matrix Heatmap:       
    
    sns.set(font_scale=1)#for label size
    ax = sns.heatmap(conmatrix_df, annot=True,annot_kws={"size": 12}, fmt='d')
    ax.set_title('Confusion Matrix')
    ax.set(xlabel='Predictions', ylabel='True Labeled Values')
    plt.show()
    
    #Accuracy Loop:
    accuracy_set = []
    
    for x in range(0,1000,1):
        test_size = 0.50
        X_train, X_test, y_train, y_test = train_test_split(train_colors, ytarget, test_size=test_size)

        model = XGBClassifier()
        model.fit(X_train, y_train)
        #make predictions for test data
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        #print("Accuracy: %.10f%%" % (accuracy * 100.0))
        accuracy_set = np.append(accuracy_set, accuracy)
    #print(accuracy_set)
    avg_accuracy = np.mean(accuracy_set)
    sigma = np.std(accuracy_set)
    print("Accuracy Standard Deviation: %.8f" % sigma)
    print("Average Accuracy: %.10f%%" % (avg_accuracy*100))
    #print(conmatrix)
    #print(predictions1)
    #print(Counter(predictions1))
    print(Counter(test_predictions))
    
    plt.hist(accuracy_set, histtype = 'step',bins = 500, color = 'k')
    plt.xlabel('Accuracy Score')
    plt.ylabel('Frequency of Occurence')
    plt.title('Accuracy Distribution')
    plt.show()
    
XGBooster(train,skymapper)