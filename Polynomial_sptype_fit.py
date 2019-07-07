#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 14:33:18 2018

@author: Helios
"""

import matplotlib.pyplot as plt
import pandas as pd
from pylab import rcParams
from matplotlib.pyplot import figure
from collections import *
import numpy as np
import seaborn as sns
import sys
#%matplotlib inline
#sys.setrecursionlimit(10000)
#rcParams['figure.figsize'] = 7, 5

Dwarfs = pd.read_csv('/Users/Helios/Desktop/Research/CoolStars_Data/Dwarfs_n2.csv')
#print(Dwarfs['abs_spt'])
#To do: change all the absolute sptype strings into integers, then run the plot below.
#To do: make a polynomial fit using lmplot. Get the equation to predict sptype value based on i - z value.


#The integers are added so that the boxplot understands what to plot.
sps = Dwarfs['abs_spt']
D1 = sps.replace([sps[sps.str.contains('M0')]], '0') #simplify these and automate the process
D2 = D1.replace([D1[D1.str.contains('M1')]], '1')
D3 = D2.replace([D2[D2.str.contains('M2')]], '2')
D4 = D3.replace([D3[D3.str.contains('M3')]], '3')
D5 = D4.replace([D4[D4.str.contains('M4')]], '4')
D6 = D5.replace([D5[D5.str.contains('M5')]], '5')
D7 = D6.replace([D6[D6.str.contains('M6')]], '6')
D8 = D7.replace([D7[D7.str.contains('M7')]], '7')
D9 = D8.replace([D8[D8.str.contains('M8')]], '8')
D10 = D9.replace([D9[D9.str.contains('M9')]], '9')

D11 = D10.replace([D10[D10.str.contains('L0')]], '10')
D12 = D11.replace([D11[D11.str.contains('L1')]], '11')
D13 = D12.replace([D12[D12.str.contains('L2')]], '12')
D14 = D13.replace([D13[D13.str.contains('L3')]], '13')
D15 = D14.replace([D14[D14.str.contains('L4')]], '14')
D16 = D15.replace([D15[D15.str.contains('L5')]], '15')
D17 = D16.replace([D16[D16.str.contains('L6')]], '16')
D18 = D17.replace([D17[D17.str.contains('L7')]], '17')
D19 = D17.replace([D17[D17.str.contains('L8')]], '18')
D20 = D17.replace([D17[D17.str.contains('L9')]], '19')

print(D17)
Counter(D17)

Dwarfs.insert(loc=0, column='num_spt', value=D17) #changed here
Dwarfs.drop(Dwarfs.tail(2).index,inplace=True)
print(Dwarfs['num_spt'])

izspt = Dwarfs[['num_spt','i_psf', 'z_psf']]
izspt = izspt.dropna(how='any')
izspt = izspt.apply(pd.to_numeric)
izspt.dtypes
i_z = izspt['i_psf'] - izspt['z_psf']
izspt.insert(loc=0, column='i - z', value=i_z)

izspt.to_csv('/Users/Helios/Desktop/Research/CoolStars_Data/boxplot_data.csv')

def sptpred(x):
    spt = (1/0.1035)*(x - 0.2393)
    return spt

izspt_spts = sptpred(izspt['i - z'])

izspt.insert(loc=0, column='cont_spt', value=izspt_spts)

sptypes = izspt[['num_spt','cont_spt']]
print(sptypes)
Counter(sptypes['num_spt'])

y = sptypes.groupby('num_spt', as_index=False)['cont_spt'].median()
print(y['cont_spt'])#re-input these averages into the inverse of sptpred to get the color values needed for the graph

def izpred(x):
    iz = 0.1035*x + 0.2393
    return iz

izs = izpred(y['cont_spt'])
print(izs.to_string(index=False))

