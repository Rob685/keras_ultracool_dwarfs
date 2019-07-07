# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#see console 1/a
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
plt.rcParams.update(plt.rcParamsDefault)

acam = pd.read_csv('/Users/Helios/Desktop/Research/CoolStars_Data/All_Dwarfs_colors.csv', dtype=None, index_col=False)
del acam['Unnamed: 0']#All colors and magnitudes
ggiants = pd.read_csv('/Users/Helios/Desktop/Research/CoolStars_Research/GS2A1_n2.csv')
sgiants = pd.read_csv('/Users/Helios/Desktop/Research/CoolStars_Research/Giants_n.csv')
list(sgiants)

Gaia_Giants = ggiants[['ra', 'dec', 'u_psf', 'g_psf', 'r_psf', 'i_psf', 'z_psf', 'j_m', 'h_m', 'k_m', 'w1mpro', 'w2mpro']]
Gaia_Giants.rename(columns={'ra': 'raj2000', 'dec': 'dej2000'}, inplace=True)

SIMBAD_Giants = sgiants[['raj2000', 'dej2000', 'u_psf', 'g_psf', 'r_psf', 'i_psf', 'z_psf', 'j_m', 'h_m', 'k_m', 'w1mpro', 'w2mpro']]

all_giants = pd.concat([Gaia_Giants, SIMBAD_Giants])

#Building Giants list

#UGRIZ Filter color combinations
gi_z = all_giants['i_psf'] - all_giants['z_psf']
gr_i = all_giants['r_psf'] - all_giants['i_psf']
gg_r = all_giants['g_psf'] - all_giants['r_psf']
gu_g = all_giants['u_psf'] - all_giants['g_psf']
gg_i = all_giants['g_psf'] - all_giants['i_psf']
gr_z = all_giants['r_psf'] - all_giants['z_psf']

#2MASS Color combinations
gj_h = all_giants['j_m'] - all_giants['h_m']
gh_k = all_giants['h_m'] - all_giants['k_m']
gj_k = all_giants['j_m'] - all_giants['k_m']
gi_k = all_giants['i_psf'] - all_giants['k_m']


#WISE Color
gw1_w2 = all_giants['w1mpro'] - all_giants['w2mpro']
gz_w1 = all_giants['z_psf'] - all_giants['w1mpro']

all_giants.insert(loc=1, column='g - i', value=gg_i)
all_giants.insert(loc=2, column='r - i', value=gr_i)
all_giants.insert(loc=3, column='i - k', value=gi_k)
all_giants.insert(loc=1, column='g - r', value=gg_r)
all_giants.insert(loc=2, column='i - z', value=gi_z)
all_giants.insert(loc=3, column='u - g', value=gu_g)
all_giants.insert(loc=3, column='r - z', value=gr_z)

all_giants.insert(loc=3, column='j - h', value=gj_h)
all_giants.insert(loc=3, column='h - k', value=gh_k)
all_giants.insert(loc=3, column='j - k', value=gj_k)

all_giants.insert(loc=3, column='w1 - w2', value=gw1_w2)
all_giants.insert(loc=3, column='z - w1', value=gz_w1)
list(all_giants)

all_giants.insert(loc=0, column='StarType', value='Giant')
all_giants.to_csv('/Users/Helios/Desktop/Research/CoolStars_Data/All_Giants_colors.csv') #For later use

MLT = pd.concat([acam, all_giants])
print(MLT['dej2000'])
list(MLT)
MLT.to_csv('/Users/Helios/Desktop/Research/CoolStars_Data/MachineLearningTrainingSet3.csv')



ikcolumns = acam[['i - k', 'g - i']]
ikcc = ikcolumns.dropna(how='any')


ikplot = plt.scatter(ikcc['g - i'], ikcc['i - k'], s = 1, alpha = 0.3, color = 'k')
plt.title('I - K v. G - I: SkyMapper with 2MASS')
plt.ylabel(r'I - K')
plt.xlabel(r'G - I')
plt.show()

#SkyMapper 2d histogram
ikhist = plt.hist2d(ikcc['g - i'], ikcc['i - k'], bins=100, cmap=plt.cm.Greys, norm=LogNorm(), range=[[0,6],[0,8]])
plt.title('I - K v. G - I: SkyMapper with 2MASS Histogram')
plt.ylabel(r'I - K')
plt.xlabel(r'G - I')
plt.colorbar()
plt.show()
##############################################################################
#Comparison to SIMBAD list
dg_i = sdwarfs['g_psf'] - sdwarfs['i_psf']
di_k = sdwarfs['i_psf'] - sdwarfs['k_m']

gg_i = sgiants['g_psf'] - sgiants['i_psf']
gi_k = sgiants['i_psf'] - sgiants['k_m']

#SIMBAD Dwarfs and Giants
ikdplot = plt.scatter(dg_i, di_k, s = 1, alpha = 0.9, color = 'r')
ikgplot = plt.scatter(gg_i, gi_k, s = 1, alpha = 0.9, color = 'b')
plt.title('I - K v. G - I: SkyMapper with 2MASS')
plt.ylabel(r'I - K')
plt.xlabel(r'G - I')
plt.legend((ikdplot, ikgplot, ikplot),('Known Dwarfs', 'Known Giants','SkyMapper Data'), loc='upper left', markerscale = 3)
plt.show()

#SIMBAD Dwarfs and Giants compared to SM 2D Histogram
ikdplot = plt.scatter(dg_i, di_k, s = 1, alpha = 0.2, color = 'r')
ikgplot = plt.scatter(gg_i, gi_k, s = 3, alpha = 0.7, color = 'b')
ikhist = plt.hist2d(ikcc['g - i'], ikcc['i - k'], bins=100, cmap=plt.cm.Greys, norm=LogNorm(), range=[[0,6],[0,8]])
plt.title('I - K v. G - I: Overlay')
plt.ylabel(r'I - K')
plt.xlabel(r'G - I')
plt.legend((ikdplot, ikgplot, ikplot),('Known Dwarfs', 'Known Giants','SkyMapper Data Density'), loc='upper left', markerscale = 3)
plt.colorbar()
plt.show()
##############################################################################
#The goal here is to overlay the skymapper histograms to the SIMBAD scatterplots
#SkyMapper R-Z and I-Z Graphs

#W1 - W2 v. R - Z
w1w2columns = acam[['w1 - w2', 'r - z']]
wwcc = w1w2columns.dropna(how='any')

w1w2plot = plt.scatter(wwcc['r - z'], wwcc['w1 - w2'], s = 1, alpha = 0.3, color = 'k')
plt.title('W1 - W2 v. R - Z: SkyMapper and WISE')
plt.ylabel(r'W1 - W2')
plt.xlabel(r'R - Z')
plt.show()

w1w2hist = plt.hist2d(wwcc['r - z'], wwcc['w1 - w2'], bins=100, cmap=plt.cm.Greys, norm=LogNorm(), range=[[0,6.5], [-2.2,2]])
plt.title('W1 - W2 v. R - Z: SkyMapper and WISE Histogram')
plt.ylabel(r'W1 - W2')
plt.xlabel(r'R - Z')
plt.colorbar()
plt.show()

#W1 - W2 v. I - Z

w1w2columns2 = acam[['w1 - w2', 'i - z']]
wwcc2 = w1w2columns2.dropna(how='any')

w1w2plot2 = plt.scatter(wwcc2['i - z'], wwcc2['w1 - w2'], s = 1, alpha = 0.3, color = 'k')
plt.title('W1 - W2 v. I - Z: SkyMapper and WISE')
plt.ylabel(r'W1 - W2')
plt.xlabel(r'I - Z')
plt.show()

w1w2hist2 = plt.hist2d(wwcc2['i - z'], wwcc2['w1 - w2'], bins=100, cmap=plt.cm.Greys, norm=LogNorm(), range=[[0,3], [-2.5,2]])
plt.title('W1 - W2 v. I - Z: SkyMapper and WISE Histogram')
plt.ylabel(r'W1 - W2')
plt.xlabel(r'I - Z')
plt.colorbar()
plt.show()

#####SIMBAD Dwarfs and Giants for R-Z and I-Z Graphs:
#R-Z overlay
dw1_w2 = sdwarfs['w1mpro'] - sdwarfs['w2mpro']
dr_z = sdwarfs['r_psf'] - sdwarfs['z_psf']
di_z = sdwarfs['i_psf'] - sdwarfs['z_psf']

gw1_w2 = sgiants['w1mpro'] - sgiants['w2mpro']
gr_z = sgiants['r_psf'] - sgiants['z_psf']
gi_z = sgiants['i_psf'] - sgiants['z_psf']
#plots
ww1dplot = plt.scatter(dr_z, dw1_w2, s = 1, alpha = 0.2, color = 'b')#dataplot
ww1gplot = plt.scatter(gr_z, gw1_w2, s = 1, alpha = 0.9, color = 'r')
plt.title('W1 - W2 v. R - Z: Test Samples')
plt.ylabel(r'W1 - W2')
plt.xlabel(r'R - Z')
plt.legend((ww1dplot, ww1gplot),('Known Dwarfs', 'Known Giants'), loc='lower right', markerscale = 7)
plt.show()

ww1dplot = plt.scatter(dr_z, dw1_w2, s = 1, alpha = 0.2, color = 'r')
ww1gplot = plt.scatter(gr_z, gw1_w2, s = 3, alpha = 0.9, color = 'b')
w1w2hist = plt.hist2d(wwcc['r - z'], wwcc['w1 - w2'], bins=100, cmap=plt.cm.Greys, norm=LogNorm(), range=[[0,6.5], [-2.2,2]])#overlayed plot
plt.title('W1 - W2 v. R - Z: Overlay')
plt.ylabel(r'W1 - W2')
plt.xlabel(r'R - Z')
plt.legend((ww1dplot, ww1gplot, w1w2plot),('Known Dwarfs', 'Known Giants', 'SkyMapper Data Density'), loc='lower right', markerscale = 3)
plt.colorbar()
plt.show()

#I-Z overlay

ww1dplot2 = plt.scatter(di_z, dw1_w2, s = 1, alpha = 0.2, color = 'b')#dataplot
ww1gplot2 = plt.scatter(gi_z, gw1_w2, s = 1, alpha = 0.9, color = 'r')
plt.title('W1 - W2 v. I - Z: Test Samples')
plt.ylabel(r'W1 - W2')
plt.xlabel(r'I - Z')
plt.legend((ww1dplot2, ww1gplot2),('Known Dwarfs', 'Known Giants'), loc='lower right', markerscale = 7)
plt.show()

ww1dplot2 = plt.scatter(di_z, dw1_w2, s = 1, alpha = 0.2, color = 'b')
ww1gplot2 = plt.scatter(gi_z, gw1_w2, s = 3, alpha = 0.9, color = 'r')
w1w2hist2 = plt.hist2d(wwcc2['i - z'], wwcc2['w1 - w2'], bins=100, cmap=plt.cm.Greys, norm=LogNorm(), range=[[0,3], [-2.5,2]])#overlayed plot
plt.title('W1 - W2 v. I - Z: Overlay')
plt.ylabel(r'W1 - W2')
plt.xlabel(r'I - Z')
plt.legend((ww1dplot2, ww1gplot2, w1w2plot2),('Known Dwarfs', 'Known Giants', 'SkyMapper Data Density'), loc='lower right', markerscale = 3)
plt.colorbar()
plt.show()

############################################################################
#Other greyscales to show for "shock" slide"...


ugcolumns = acam[['u - g', 'g - i']]
ugcc = ugcolumns.dropna(how='any')


ughist = plt.hist2d(ugcc['g - i'], ugcc['u - g'], bins=100, cmap=plt.cm.Greys, norm=LogNorm(), range=[[0,6],[0,4]])
plt.title('U - G v. G - I: SkyMapper with 2MASS and WISE Histogram')
plt.ylabel(r'U - G')
plt.xlabel(r'G - I')
plt.colorbar()
plt.show()


jhcolumns = acam[['j - h', 'w1 - w2']]
jhcc = jhcolumns.dropna(how='any')


jhhist = plt.hist2d(jhcc['w1 - w2'], jhcc['j - h'], bins=100, cmap=plt.cm.Greys, norm=LogNorm(), range=[[-2,2],[0,2]])
plt.title('J - H v. W1 - W2: SkyMapper with 2MASS and WISE Histogram')
plt.ylabel(r'J - H')
plt.xlabel(r'W1 - W2 (mag)')
plt.colorbar()
plt.show()

zw1columns = acam[['z - w1', 'g - r']]
zw1cc = zw1columns.dropna(how='any')

zw1hist = plt.hist2d(zw1cc['g - r'], zw1cc['z - w1'], bins=100, cmap=plt.cm.Greys, norm=LogNorm(), range=[[-0.5,3],[0,7]])
plt.title('Z - W1 v. G - R: SkyMapper with 2MASS and WISE Histogram')
plt.ylabel(r'Z - W1')
plt.xlabel(r'G - R')
plt.colorbar()
plt.show()

