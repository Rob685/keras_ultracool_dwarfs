#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 11:45:50 2018

@author: roberttejada
"""

import matplotlib.pyplot as plt
import pandas as pd
from pylab import rcParams
from matplotlib.pyplot import figure

rcParams['figure.figsize'] = 10, 5

m = pd.read_csv('/Users/roberttejada/Desktop/CoolStars_Data/Dwarflocs_Skymapper_matches2.0.csv', delimiter = ',')

sptypes = ['M0', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'L0', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9']

colors = ['blue', 'skyblue', 'darkcyan', 'mediumspringgreen', 'seagreen', 'darkgreen', 'palegreen', 'olivedrab', 'gold', 'orange', 'coral', 'lightsalmon', 'salmon', 'lightcoral', 'indianred', 'orangered', 'red', 'firebrick', 'brown', 'maroon']

#x = m[m['col3'].str.contains('%s'%sptypes[0])]

#r_i = x['r_psf'] - x['i_psf']
#i_z = x['i_psf'] - x['z_psf']
#g_r = x['g_psf'] - x['r_psf']


#Dwarf Match Plots with 6" Search Radius:

m = pd.read_csv('/Users/roberttejada/Desktop/CoolStars_Data/Dwarflocs_Skymapper_matches2.0_6r.csv', delimiter = ',')

for i in range(len(sptypes)):
    x = m[m['col3'].str.contains('%s'%sptypes[i])]
    r_i = x['r_psf'] - x['i_psf']
    i_z = x['i_psf'] - x['z_psf']
    ax1 = plt.scatter(i_z, r_i, color=colors[i], alpha = 0.4, label = '%s'%sptypes[i])
plt.xlabel(r'i-z')
plt.ylabel(r'r-i')
plt.title('SIMBAD and SkyMapper X-Match: 6" Search Radius')
plt.legend(loc='best', fontsize=8.5)
plt.show()

for i in range(len(sptypes)):
    x = m[m['col3'].str.contains('%s'%sptypes[i])]
    r_i = x['r_psf'] - x['i_psf']
    g_r = x['g_psf'] - x['r_psf']
    ax1 = plt.scatter(r_i, g_r, color=colors[i], alpha = 0.4, label = '%s'%sptypes[i])
plt.xlabel(r'r-i')
plt.ylabel(r'g-r')
plt.title('SIMBAD and SkyMapper X-Match: 6" Search Radius')
plt.legend(loc='best', fontsize=8.5)
plt.show()

#Dwarf Match plots with 3" Search Radius:

n = pd.read_csv('/Users/roberttejada/Desktop/CoolStars_Data/Dwarflocs_Skymapper_matches2.0_3r.csv', delimiter = ',')

for i in range(len(sptypes)):
    y = n[n['col3'].str.contains('%s'%sptypes[i])]
    r_i = y['r_psf'] - y['i_psf']
    i_z = y['i_psf'] - y['z_psf']
    ax2 = plt.scatter(i_z, r_i, color=colors[i], alpha = 0.4, label = '%s'%sptypes[i])
plt.xlabel(r'i-z')
plt.ylabel(r'r-i')
plt.title('SIMBAD and SkyMapper X-Match: 3" Search Radius')
plt.legend(loc='best', fontsize=8.5)
plt.show()

for i in range(len(sptypes)):
    y = n[n['col3'].str.contains('%s'%sptypes[i])]
    r_i = y['r_psf'] - y['i_psf']
    g_r = y['g_psf'] - y['r_psf']
    ax2 = plt.scatter(r_i, g_r, color=colors[i], alpha = 0.4, label = '%s'%sptypes[i])
plt.xlabel(r'r-i')
plt.ylabel(r'g-r')
plt.title('SIMBAD and SkyMapper X-Match: 3" Search Radius')
plt.legend(loc='best', fontsize=8.5)
plt.show()

#Dwarf Match plots with 12" Search Radius (Flags=0 Working Plots):

t = pd.read_csv('/Users/roberttejada/Desktop/CoolStars_Data/Dwarflocs_Skymapper_matches3.0_12r.csv', delimiter = ',')

for i in range(len(sptypes)):
    z = t[t['col3'].str.contains('%s'%sptypes[i])]
    r_i = z['r_psf'] - z['i_psf']
    i_z = z['i_psf'] - z['z_psf']
    ax3 = plt.scatter(i_z, r_i, color=colors[i], s = 8, alpha = 0.4, label = '%s'%sptypes[i])
plt.xlabel(r'i-z')
plt.ylabel(r'r-i')
plt.title('SIMBAD and SkyMapper X-Match: 12" Search Radius')
plt.legend(loc='best', fontsize=8.5)
plt.show()

for i in range(len(sptypes)):
    z = t[t['col3'].str.contains('%s'%sptypes[i])]
    r_i = z['r_psf'] - z['i_psf']
    g_r = z['g_psf'] - z['r_psf']
    ax3 = plt.scatter(r_i, g_r, color=colors[i], s = 8, alpha = 0.4, label = '%s'%sptypes[i])
plt.xlabel(r'r-i')
plt.ylabel(r'g-r')
plt.title('SIMBAD and SkyMapper X-Match: 12" Search Radius')
plt.legend(loc='best', fontsize=8.5)
plt.show()



################################################################
#Giant Match plots with 6" Search Radius

g = pd.read_csv('/Users/roberttejada/Desktop/CoolStars_Data/Giantlocs_Skymapper_matches2.0_6r.csv', delimiter = ',')

for i in range(len(sptypes)):
    p = g[g['col3'].str.contains('%s'%sptypes[i])]
    r_i = p['r_psf'] - p['i_psf']
    i_z = p['i_psf'] - p['z_psf']
    ax4 = plt.scatter(i_z, r_i, color=colors[i], alpha = 0.4, label = '%s'%sptypes[i])
plt.xlabel(r'i-z')
plt.ylabel(r'r-i')
plt.title('SIMBAD and SkyMapper X-Match')
plt.legend(loc='best', fontsize=8.5)
plt.show()

for i in range(len(sptypes)):
    p = g[g['col3'].str.contains('%s'%sptypes[i])]
    r_i = p['r_psf'] - p['i_psf']
    g_r = p['g_psf'] - p['r_psf']
    ax2 = plt.scatter(r_i, g_r, color=colors[i], alpha = 0.4, label = '%s'%sptypes[i])
plt.xlabel(r'r-i')
plt.ylabel(r'g-r')
plt.title('SIMBAD and SkyMapper X-Match')
plt.legend(loc='best', fontsize=8.5)
plt.show()


##############################################################################################
#Giants and Dwarfs Plots: Dwarf Locs - SkyMapper X-Match

gr_i = g['r_psf'] - g['i_psf']#Giant colors
gi_z = g['i_psf'] - g['z_psf']
gg_r = g['g_psf'] - g['r_psf']

dr_i = t['r_psf'] - t['i_psf']#Dwarf colors
di_z = t['i_psf'] - t['z_psf']
dg_r = t['g_psf'] - t['r_psf']

A = plt.scatter(gr_i, gg_r, c = 'g', s = 8, alpha = 0.8)
B = plt.scatter(dr_i, dg_r, c = 'r', s = 8, alpha = 0.2)
plt.xlabel(r'r-i')
plt.ylabel(r'g-r')
#plt.xlim(0, 3.7)
#plt.ylim(0, 3.7)
plt.title('G - R v. R - I: Giants and Dwarfs')
plt.legend((A, B),('Giants', 'Dwarfs'))
plt.show()

C = plt.scatter(gi_z, gr_i, c = 'g', s = 8, alpha = 0.8)
D = plt.scatter(di_z, dr_i, c = 'r', s = 8, alpha = 0.2)
plt.xlabel(r'i-z')
plt.ylabel(r'r-i')
#plt.xlim(0, 3.7)
#plt.ylim(0, 3.7)
plt.title('R - I v. I - Z: Giants and Dwarfs')
plt.legend((C, D),('Giants', 'Dwarfs'))
plt.show()



