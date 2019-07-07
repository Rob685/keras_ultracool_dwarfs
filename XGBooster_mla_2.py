#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 14:40:14 2018

@author: Helios
"""
#See console3/a
import pandas as pd
import numpy as np
from collections import Counter
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from itertools import combinations
import seaborn as sns
import warnings
warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
from pylab import rcParams
rcParams['figure.figsize'] = 7,5


train = pd.read_csv('/Users/Helios/Desktop/Research/CoolStars_Data/XGBoost_Classifier_TSET.csv', index_col=0)#Training labeled set
list(train)
Counter(train['main_type'])

train = train.replace('Giant','Contaminant')

Counter(train['StarType'])

uobjs = pd.read_csv('/Users/Helios/Desktop/Research/CoolStars_Data/unmatched_objs.csv', index_col=0)#Gaia unmached objs
list(uobjs)

#Target data (SkyMapper):
skymapper = pd.read_csv('/Users/Helios/Desktop/Research/CoolStars_Data/xmatch_colors_wuncert_new.csv', index_col=0)
list(skymapper)


#Cleaning:
tmags = train[['i_psf', 'z_psf', 'h_m', 'j_m', 'k_m', 'w1mpro','w2mpro', 'StarType']]
list(tmags)
Counter(train['StarType'])
tmags2 = tmags.dropna(how='any')
Counter(tmags2['StarType'])
tmags3 = tmags2[['i_psf', 'z_psf', 'h_m', 'j_m', 'k_m', 'w1mpro','w2mpro']]

#Targets:
umags = uobjs[['i_psf', 'z_psf', 'h_m', 'j_m', 'k_m', 'w1mpro', 'w2mpro']]#Unmatched objects
umags2 = umags.dropna(how='any')
smmags = skymapper[['i_psf', 'z_psf', 'h_m', 'j_m', 'k_m', 'w1mpro', 'w2mpro']]#all skymapper
smmags2 = smmags.dropna(how='any')

#Combination function:
def ccombinator(x):
    mags1 = pd.DataFrame(index=x.index)
    mags2 = pd.DataFrame(index=x.index)
    for a, b, in combinations(x.columns, 2):
        mags1['{}-{}'.format(a, b)] = x[a] - x[b]
        mags2['{}-{}'.format(b, a)] = x[b] - x[a]
    c = mags1.join(mags2)
    return c

tcolors = ccombinator(tmags3).join(tmags3, how='outer')
list(tcolors)
ucolors = ccombinator(umags2).join(umags2, how='outer')#These are the new objects
smcolors = ccombinator(smmags2).join(smmags2, how='outer')

ytarget = tmags2.StarType

#Split data into train and test sets
#seed = 45

test_size = 0.50
X_train, X_test, y_train, y_test = train_test_split(tcolors, ytarget, test_size=test_size)
    
#fit model no training data
    
model = XGBClassifier()
model.fit(X_train, y_train)
    
#make predictions for test data
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.10f%%" % (accuracy * 100.0))
    
#Predictions:
conmatrix = confusion_matrix(y_test,predictions)
print(conmatrix)
print(predictions)
Counter(predictions)


conmatrix_df = pd.DataFrame.from_items([('Contaminants', conmatrix[0]), ('Dwarfs', conmatrix[1])], columns=['Pred. Contaminants', 'Pred. Dwarfs'], orient='index')
print(conmatrix_df)

#Confusion Matrix Heatmap:       

sns.set(font_scale=1)#for label size
ax = sns.heatmap(conmatrix_df, annot=True,annot_kws={"size": 12}, fmt='d')
ax.set_title('Confusion Matrix')
ax.set(xlabel='Predictions', ylabel='True Labeled Values')
plt.show()


#uobjs predictions:

rpredictions = model.predict(ucolors)
Counter(rpredictions)

smpredictions = model.predict(smcolors)
Counter(smpredictions)

#Feature Importance Plot:
plot_importance(model)
plt.show()

############################################################
#Accuracy Iteration Loop:
accuracy_dist = np.array([])
counter_list = []

for x in range(0,1000,1):
    test_size = 0.50
    X_train, X_test, y_train, y_test = train_test_split(tcolors, ytarget, test_size=test_size)
    
    #fit model w/ training data
    
    model = XGBClassifier()
    model.fit(X_train, y_train)
    
    #make predictions for test data
    
    y_pred = model.predict(X_test)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    #print("Accuracy: %.10f%%" % (accuracy * 100.0))
    accuracy_dist = np.append(accuracy_dist, accuracy)
print(accuracy_dist)
sigma = np.std(accuracy_dist)
print(sigma)

mean_acc = np.mean(accuracy_dist)
print(mean_acc)

############################################################
#Plots:
plt.hist(accuracy_dist, histtype = 'step',bins = 1000, color = 'k')
plt.xlabel('Accuracy')
plt.ylabel('Frequency of Occurence')
plt.title('Distribution of Accuracy')
plt.show()

#2dhist of most important predictor:
sns.set_style("whitegrid", {'axes.grid' : False})
plt.hist2d(tcolors['i_psf-z_psf'], tcolors['w1mpro-w2mpro'], bins=100, cmap=plt.cm.Greys, norm=LogNorm(),range=[[-0.5,2.0],[-0.5,1.0]])
plt.title('w1 - w2 v. i - z: Training Set')
plt.ylabel(r'w1 - w2')
plt.xlabel(r'i - z')
plt.colorbar().set_label('# of Objects')
plt.show()

plt.hist2d(ucolors['i_psf-z_psf'], ucolors['w1mpro-w2mpro'], bins=100, cmap=plt.cm.Greys, norm=LogNorm(),range=[[-0.5,2.0],[-0.5,1.0]])
plt.title('w1 - w2 v. i - z: Unmatched Set')
plt.ylabel(r'w1 - w2')
plt.xlabel(r'i - z')
plt.colorbar().set_label('# of Objects')
plt.show()


iz = train[['i - z','r - i']]
iz_ri = iz.dropna(how='any')

iz2 = skymapper[['i - z','r - i']]
iz_ri2 = iz2.dropna(how='any')

#training set comparisons with skymapper set
plt.hist2d(iz_ri['i - z'], iz_ri['r - i'], bins=1000, norm=LogNorm(),cmap=plt.cm.viridis)
plt.xlabel(r'i - z')
plt.ylabel(r'r - i')
plt.colorbar()
plt.show()

plt.hist2d(iz_ri2['i - z'], iz_ri2['r - i'], bins=1000, norm=LogNorm(),cmap=plt.cm.viridis,range=[[-0.5,2.5],[-0.2,4]])
plt.xlabel(r'i - z')
plt.ylabel(r'r - i')
plt.colorbar()
plt.show()


#######################################################################

sns.set_style("whitegrid", {'axes.grid' : False})
plt.hist2d(smcolors['i_psf-z_psf'], smcolors['w1mpro-w2mpro'], bins=100, cmap=plt.cm.Greys, norm=LogNorm(), range=[[0.25,1.5],[-1, 1.5]]
           )
plt.title('w1 - w2 v. i - z: SkyMapper Set')
plt.ylabel(r'w1 - w2')
plt.xlabel(r'i - z')
plt.colorbar().set_label('# of Objects')
plt.show()

#Scatter plot of w1 - w2 v. i - z:
tcont = tmags.loc[tmags['StarType'] == 'Contaminant']
tdwarfs = tmags.loc[tmags['StarType'] == 'Dwarf']
tcont_colors = tcont[['i_psf', 'z_psf', 'h_m', 'j_m', 'k_m', 'w1mpro','w2mpro']]
tdwarfs_colors = tdwarfs[['i_psf', 'z_psf', 'h_m', 'j_m', 'k_m', 'w1mpro','w2mpro']]

tcolorsn = ccombinator(tcont_colors)
tdwarfsn = ccombinator(tdwarfs_colors)
list(tcolorsn)

print(tcont)

sns.set()
ax2 = plt.scatter(tdwarfsn['i_psf-z_psf'], tdwarfsn['w1mpro-w2mpro'], s = 1, c = 'r', alpha = 0.2, norm = LogNorm())
ax1 = plt.scatter(tcolorsn['i_psf-z_psf'],tcolorsn['w1mpro-w2mpro'], s = 1, c = 'k', alpha=0.2,  norm = LogNorm())
plt.title(r'w1 - w2 v. i - z: Dwarfs and Other Objects')
plt.xlabel(r'i - z')
plt.ylabel(r'w1 - w2')
plt.xlim(-0.5,2.0)
plt.ylim(-0.5,1.0)
plt.legend((ax1, ax2), ('Other Objects', 'Dwarfs'), markerscale=8)
plt.show()

sns.set()
ax2 = plt.hist2d(tdwarfsn['i_psf-z_psf'], tdwarfsn['w1mpro-w2mpro'], bins=100, cmap=plt.cm.Greys, norm=LogNorm(),range=[[-0.5,2.0],[-0.5,1.0]])
ax1 = plt.hist2d(tcolorsn['i_psf-z_psf'], tcolorsn['w1mpro-w2mpro'], bins=100, cmap=plt.cm.Purples, norm=LogNorm(),range=[[-0.5,2.0],[-0.5,1.0]])
plt.title('w1 - w2 v. i - z: Training Set')
plt.ylabel(r'w1 - w2')
plt.xlabel(r'i - z')
plt.colorbar().set_label('# of Objects')
plt.show()


ax3 = plt.scatter(smcolors['i_psf-z_psf'], smcolors['w1mpro-w2mpro'], s = 1, alpha = 0.2, c = 'k', norm=LogNorm())
ax4 = plt.scatter(ucolors['i_psf-z_psf'], ucolors['w1mpro-w2mpro'], s = 1, alpha = 0.3, c = 'r')
plt.title('w1 - w2 v. i - z')
plt.ylabel('w1 - w2')
plt.xlabel('i - z')
plt.xlim(0.25,1.6)
plt.ylim(-2,3)
plt.legend((ax3, ax4), ('Skymapper and Gaia', 'SkyMapper objects unmached with Gaia'), markerscale=6.5)
plt.show()



#No need:

###################################################################
#Without 2MASS data:
#Cleaning:
tmags_n = train[['i_psf', 'z_psf', 'w1mpro','w2mpro', 'StarType']]
#print(X3)
#Counter(train['StarType'])
tmags2_n = tmags.dropna(how='any')
Counter(tmags2_n['StarType'])
tmags3_n = tmags2_n[['i_psf', 'z_psf', 'w1mpro','w2mpro']]

#Targets:
umags = uobjs[['i_psf', 'z_psf', 'w1mpro', 'w2mpro']]#Unmatched objects
umags2 = umags.dropna(how='any')
smmags = skymapper[['i_psf', 'z_psf', 'w1mpro', 'w2mpro']]#all skymapper
smmags2 = smmags.dropna(how='any')

#Combination function:
def ccombinator(x):
    mags1 = pd.DataFrame(index=x.index)
    mags2 = pd.DataFrame(index=x.index)
    for a, b, in combinations(x.columns, 2):
        mags1['{}-{}'.format(a, b)] = x[a] - x[b]
        mags2['{}-{}'.format(b, a)] = x[b] - x[a]
    c = mags1.join(mags2)
    return c

tcolors = ccombinator(tmags3)
ucolors = ccombinator(umags2)
smcolors = ccombinator(smmags2)


ytarget = tmags2.StarType

#Split data into train and test sets
#seed = 45

test_size = 0.50
X_train, X_test, y_train, y_test = train_test_split(tcolors, ytarget, test_size=test_size)
    
#fit model no training data
    
model = XGBClassifier()
model.fit(X_train, y_train)
    
#make predictions for test data
    
y_pred = model.predict(X_test)
predictions2 = model.predict(X_test)
accuracy2 = accuracy_score(y_test, predictions2)
print("Accuracy: %.10f%%" % (accuracy * 100.0))

#Feature Importance Plot:
plot_importance(model)
plt.show()

#Accuracy Loop without 2MASS mags:
accuracy_dist2 = np.array([])

for x in range(0,1000,1):
    test_size = 0.50
    X_train, X_test, y_train, y_test = train_test_split(tcolors, ytarget, test_size=test_size)
    
    #fit model no training data
    
    model = XGBClassifier()
    model.fit(X_train, y_train)
    
    #make predictions for test data
    
    y_pred = model.predict(X_test)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    #print("Accuracy: %.10f%%" % (accuracy * 100.0))
    accuracy_dist2 = np.append(accuracy_dist2, accuracy)
print(accuracy_dist2)
sigma2 = np.std(accuracy_dist2)
print(sigma2)

mean_acc2 = np.mean(accuracy_dist2)
print(mean_acc2)

plt.hist(accuracy_dist2, histtype = 'step',bins = 1000, color = 'k')
plt.xlabel('Accuracy')
plt.ylabel('Frequency of Occurence')
plt.title('Distribution of Accuracy Without 2MASS Data')
plt.show()
