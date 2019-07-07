#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 08:36:59 2018

@author: Helios
"""
#see console3/a
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('legend', fontsize=7)
plt.rc('axes', labelsize=10)

# Set Units
d2a = 3600.
d2ma = 3600000.

'########################################################################################## STARTING'
#########################################################################################################

#t0 = Table.read('../Catalogs4/LaTeMoVeRS_v0_9_3.hdf5') # open an HDF5 file
#print t0.colnames
#print len(t0)
#print 'Number of unique matches:', len(set(t0['SDSS_OBJID'].data))

#print len(np.where( (t0['JMAG'] != -9999) & (t0['JMAG_ERR'] != -9999) )[0])
#print len(np.where( (t0['W1MPRO'] != -9999) & (t0['W1SIGMPRO'] != -9999) )[0])
#print len(np.where( (t0['W1MPRO'] != -9999) & (t0['W1SIGMPRO'] != -9999) & (t0['JMAG'] != -9999) & (t0['JMAG_ERR'] != -9999) )[0])
#sys.exit()

#t    = t0[np.where( (t0['W1MPRO'] != -9999) & (t0['W1SIGMPRO'] != -9999) & (t0['JMAG'] != -9999) & (t0['JMAG_ERR'] != -9999) )]
#print 'Length:',len(t)

#Mi   = 4.471 + 7.907*(t['IMAG']-t['ZMAG']) - 0.837*((t['IMAG']-t['ZMAG'])**2)
#dist = 10**( (t['IMAG'] - Mi) / 5. + 1)
#Mj   = t['JMAG'] + 5 - 5*np.log10(dist)

features = pd.read_csv('/Users/Helios/Desktop/Research/CoolStars_Data/xmatch_colors_new.csv', index_col=0)

features1 = features[['i - k', 'g - i']]
features1 = features1.dropna(how='any')
features2 = features[['r - z', 'w1 - w2']]
features2 = features2.dropna(how='any')
features3 = features[['i - z', 'w1 - w2']]
features3 = features3.dropna(how='any')
features4 = features[['i - z', 'i - k']]
features4 = features4.dropna(how='any')

x = features4['i - z']
#y = Mj
y = features4['i - k']
#z = t['TEMP']

beg, end = 0, 5

fig = plt.figure(1)
#ax  = fig.add_subplot(111)
#ax.scatter(x, y, s=1, alpha=0.5, label='LaTe-MoVeRS')
#xbins, ybins = np.arange(1, 3.5, .02), np.arange(-3, 5, .02)
#x1, x2 = 1, 3
#xbins, ybins = np.arange(x1, x2, .05), np.arange(0, 4, .05)
#plt.scatter(x, y, s=1, c='b', edgecolors='None', zorder=-100)#, vmax=1000)
plt.hist2d(x, y, bins=100, cmap=plt.cm.Greys, norm=LogNorm(), range=[[0.3,2],[0,6]])#, vmax=1000)
#plt.hexbin(x, y, C=z, gridsize=300)
plt.minorticks_on()
plt.xlim(0.3,2)
cbar = plt.colorbar()
cbar.set_label(r'# of Stars')
#plt.axvline(0.77, c='r', ls='--')

plt.xlabel(r'i-z')
plt.ylabel(r'i-k')

ax2 = plt.twiny() # now, ax3 is responsible for "top" axis and "right" axis
colors = [0.2780,
0.3520,
0.4275,
0.5170,
0.6110,
0.7365,
0.8960,
1.0020,
1.3625,
1.4675,
1.4075,
1.4480,
1.4380,
1.1780,
1.4280,
1.5735]
ax2.set_xticks( colors )
ax2.set_xticklabels([r"M0", r"M1", r"M2", r"M3", r"M4", r"M5", r"M6",r"M7", r"M8", r"M9", r"L0", r"L1", r"L2", r"L3", r"L4", r"L5"])
#ax2.set_yticklabels([])
ax2.set_xlim(0.3,2)

#ax.minorticks_on()
#plt.savefig('iz_JW1.png', dpi=600, bbox_inches='tight')
#plt.savefig('iz_JW1.pdf', dpi=600, bbox_inches='tight')
plt.show()

################################################################################
#find the 10 reddest objects inthe data.

