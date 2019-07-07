#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 11:13:58 2019

@author: Helios
"""
#see console 5/a
import pandas as pd
from astropy.io import ascii
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from itertools import combinations
import seaborn as sns
from collections import Counter
from pylab import rcParams
rcParams['figure.figsize'] = 10,5

mdwarfs = Table.read('/Users/Helios/Desktop/Research/SDSS_Dwarfs/mdwarfs_sdssdr7.fit')
ldwarfs = Table.read('/Users/Helios/Desktop/Research/SDSS_Dwarfs/ldwarfs_sdssdr7.fit')
mdwarfs = mdwarfs.to_pandas()
ldwarfs = ldwarfs.to_pandas()
list(mdwarfs)
print(mdwarfs['SDSS7'])
list(ldwarfs)
print(ldwarfs['DR7'])
dwarfs1 = mdwarfs[['SimbadName','SpT','RAJ2000','DEJ2000','umag','e_umag','gmag','e_gmag','rmag','e_rmag','imag','e_imag','zmag','e_zmag']]

print(dwarfs1['SpT'])
Counter(dwarfs1['SpT'])

sps = dwarfs1['SpT']
Counter(sps)
integers = range(0,10)
sptypes = ['M0', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']

#dwarfs2 = dwarfs1['SpT'].replace({0:sptypes,1:sptypes,2:sptypes,3:sptypes,4:sptypes,5:sptypes,6:sptypes,7:sptypes,8:sptypes,9:sptypes})
#Counter(dwarfs2['SpT'])

D = sps.replace(integers,sptypes)

dwarfs1['SpT'] = D

Counter(dwarfs1['SpT'])

ldwarfs1 = ldwarfs[['SimbadName','SpT','_RA','_DE','imag','e_imag','zmag','e_zmag','Jmag','e_Jmag','Hmag','e_Hmag','Ksmag','e_Ksmag']]
dwarfs2 = ldwarfs1.rename(index=str, columns={"_RA": "RAJ2000", "_DE": "DEJ2000"})

dwarfs = dwarfs1.append(dwarfs2)
list(dwarfs)

Counter(dwarfs['SpT'])


dwarfs.to_csv('/Users/Helios/Desktop/Research/SDSS_Dwarfs/combined_sdss_dwarfs2.csv')

dwarfs_sm = pd.read_csv('/Users/Helios/Desktop/Research/SDSS_Dwarfs/sdssdr7_dwarfs_sm.csv')
list(dwarfs_sm)

print(dwarfs_sm['raj2000_cone'])
print(dwarfs_sm['dej2000_cone'])

dwarfs_clean = dwarfs_sm[(dwarfs_sm['flags'] == 0)]
list(dwarfs_clean)

print(dwarfs_clean['SpT'])
dwarfs_clean.to_csv('/Users/Helios/Desktop/Research/SDSS_Dwarfs/sdssdr7_clean_sm.csv',index=False)

Counter(dwarfs_clean['SpT'])
list(dwarfs_clean)

#from coldbrew NOAO matches:
t = pd.read_csv('/Users/Helios/Desktop/Research/SDSS_Dwarfs/sdssallwise2xmatch.csv',index_col=0)
list(t)
t.columns = t.columns.str.replace('t1_','')


tcuts = t[(t['flags'] == 0) & (t['cc_flags'] == '0000') & (t['ext_flg'] == 0)]


tcuts.to_csv('/Users/Helios/Desktop/Research/SDSS_Dwarfs/sdssallwise2xmatch_clean.csv',index=False)

#Counter(t['spt'])

#tcuts['spt'] = tcuts['spt'].str.replace("b'","")
#tcuts['spt'] = tcuts['spt'].str.replace("'","")
#Counter(tcuts['spt'])#41 L objects

#####################################################################
d = pd.read_csv('/Users/Helios/Desktop/Research/SDSS_Dwarfs/sdssallwise2mass2.csv',index_col=False)

d.columns = d.columns.str.replace('t1_', '')#trimming the t1_ headers out
list(d)
print(d['spt'])
d['spt'] = d['spt'].str.replace("b'","")
d['spt'] = d['spt'].str.replace("'","")
Counter(d['spt'])

ldalwise = pd.read_csv('/Users/Helios/Desktop/Research/Ldwarfs/ldwarfs_sall2.csv',index_col=False)#supplemental dwarfs from SIMBAD....

ldalwise = ldalwise.loc[:, ~ldalwise.columns.duplicated()]

l = ldalwise.rename(index=str, columns={"abs_spt":"spt"})
list(l)


#trying to combine the ldwarfs from simbad with the sdss dr 7 dwarfs by creating new dataframes with similar columns:
sds = d[['object_id','raj2000_cone','dej2000_cone','spt','flags','cc_flags','cc_flg','ext_flg','r_psf', 'e_r_psf', 'r_petro', 'e_r_petro', 'i_psf', 'e_i_psf', 'i_petro', 'e_i_petro', 'z_psf', 'e_z_psf','w1mpro', 'w1sigmpro', 'w1snr','w1nm','w2mpro', 'w2sigmpro', 'w2snr','w2nm','j_m', 'j_cmsig', 'j_msigcom', 'j_snr', 'h_m', 'h_cmsig', 'h_msigcom', 'h_snr', 'k_m', 'k_cmsig', 'k_msigcom', 'k_snr','glon', 'glat','gal_contam']]

sds2 = sds.rename(index=str, columns={'raj2000_cone': 'raj2000', 'dej2000_cone': 'dej2000'})
list(sds2)

sds2 = sds2.loc[:, ~sds2.columns.duplicated()]

ln = l[['object_id','raj2000','dej2000','spt','flags','cc_flags','cc_flg','ext_flg','r_psf', 'e_r_psf', 'r_petro', 'e_r_petro', 'i_psf', 'e_i_psf', 'i_petro', 'e_i_petro', 'z_psf', 'e_z_psf','w1mpro', 'w1sigmpro', 'w1snr','w1nm','w2mpro','w2sigmpro','w2snr','w2nm','j_m', 'j_cmsig', 'j_msigcom', 'j_snr', 'h_m', 'h_cmsig', 'h_msigcom', 'h_snr', 'k_m', 'k_cmsig', 'k_msigcom', 'k_snr','glon', 'glat','gal_contam']]
list(ln)



comb = pd.concat([sds2,ln])
list(comb)

Counter(comb['cc_flg'])

dcuts1 = comb[(comb['flags'] == 0) & (comb['cc_flags'] == '0000') & (comb['cc_flg'] == '000') & (comb['ext_flg'] == 0)]

Counter(dcuts1['spt'])

dcuts2 = dcuts1[(dcuts1['w1snr'] >= 10) & (dcuts1['w2snr'] >= 10) & (dcuts1['j_snr'] >= 3) & (dcuts1['k_snr'] >= 5) & (dcuts1['h_snr'] >= 5) & (dcuts1['w1nm'] > 8) & (dcuts1['w2nm'] > 8)]


Counter(dcuts2['spt'])#39 L objects
print(dcuts2)

#dcuts = d[(d['flags'] == 0) & (d['cc_flags'] == 0) & (d['cc_flg'] == '000') & (d['ext_flg'] == 0) & (d['w1snr'] >= 5) & (d['w2snr'] >= 5) & (d['j_snr'] >= 3) & (d['k_snr'] >= 3) & (d['h_snr'] >= 3) & (d['w1nm'] > 8) & (d['w2nm'] > 8)]
list(dcuts2)


#Boxplot for sptypes:
sptypes2 = ['M0', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'L0', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9']
sns.boxplot(x=dcuts2["spt"], y=dcuts2["i_psf"] - dcuts2["z_psf"], order = sptypes2, palette='Greys', showfliers=False)
plt.xlabel(r'Spectral Type')
plt.ylabel(r'i - z')
plt.show()

#getting the L objects for Chris:
inv = dcuts2[dcuts2['spt'].str.contains('L3|L6|L8|L9')]
Counter(inv['spt'])

inv_s = inv[['raj2000','dej2000','spt','i_psf','z_psf']]
print(inv_s.to_string(index=False))

latels = inv[['raj2000','dej2000']] #getting the coords to search IRAS for bad photometry data (Late L's...)
t1 = Table.from_pandas(latels)
print(t1)

ascii.write([t1['raj2000'],t1['dej2000']], '/Users/Helios/Desktop/latels_coords.txt', format='ipac', names=['ra','dec'])


###############################################################

#num_spt boxplot:
sps2 = dcuts2['spt']
num_spt = range(0,20)

D2 = sps.replace(sptypes2,num_spt)
dcuts2.insert(loc=0, column='num_spt', value=D2)

izspt = dcuts2[['num_spt','i_psf', 'z_psf']]
izspt = izspt.dropna(how='all')
izspt = izspt.apply(pd.to_numeric)

x = izspt['num_spt']
x = x.values
y = izspt['i_psf'] - izspt['z_psf']
y = y.values

## Create the figure
plt.figure(1)

for i in np.unique(x):
    j = np.where(x == i)
    plt.boxplot(y[j], positions=[i], vert=True, showfliers=False)


# fit a line (first degree polynomial) to the data
z1 = np.polyfit(x, y, 1)
# fit a second degree polynomial to the data
z2 = np.polyfit(x, y, 2)

# create functions for the polynomials (we can just input the 'X' value to get the 'Y' value from the polynomail)
p1 = np.poly1d(z1)
p2 = np.poly1d(z2)
print(p1)
# Plot the polynomials
xplot = np.linspace(np.min(x),np.max(x))
plt.plot(xplot, p1(xplot), 'r-', label='1st order', alpha=0.5)
plt.plot(xplot, p2(xplot), 'm-', label='2nd order', alpha=0.5)
plt.xlim(np.min(x)-1, np.max(x)+1)
plt.ylim(np.min(y)-0.1*(np.min(y)), np.max(y)+0.1*(np.max(y)))
plt.minorticks_on()
plt.legend()
plt.xlabel('SpT')
plt.ylabel('i-z')
#plt.savefig('iz_vs_spt.png')
plt.show()