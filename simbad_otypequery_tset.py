#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 16:47:04 2019

@author: Helios
"""

import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams
import numpy as np
from collections import Counter
from astroquery.simbad import Simbad
Simbad.TIMEOUT = None
import requests
#see console 2/a
requests.get('http://simbad.u-strasbg.fr', timeout=None)

Simbad.add_votable_fields('otype','sptype')

#astroquery attempt to get data:
lowmass = Simbad.query_criteria('dec<0',otype='LM*')#lowmass stars
lms = lowmass.to_pandas()
#lms1 = lms[lms['SP_TYPE'].str.contains(b'M')]

browndwarfs = Simbad.query_criteria('dec<0',otype='BD*')#brown dwarfs
bds = browndwarfs.to_pandas()

redgiant = Simbad.query_criteria('dec<0',otype='RG*')#red giant branch star
rgb = redgiant.to_pandas()

redgals = Simbad.query_criteria('dec<0',otype='HzG')#red shifted galaxies
rgal = redgals.to_pandas()

rsuperg = Simbad.query_criteria('dec<0',otype='s*r')#Red super giants
rsuper = rsuperg.to_pandas()

train1 = [lms, bds, rgb, rsuper, rgal]

train = pd.concat(train1)
list(train)
Counter(train['SP_QUAL'])

train2 = train[train['SP_TYPE'].str.contains(b'M|L')]
Counter(train2['SP_TYPE'])
Counter(train2['OTYPE'])

train2.to_csv('/Users/Helios/Desktop/Research/CoolStars_Research/otype_tset2.csv')#otype_tset.csv was the original otype dataset without the red super giants and the red shifted galaxies.

ssm = pd.read_csv('/Users/Helios/Desktop/Research/CoolStars_Research/otypes_sim_sm2.csv', index_col=0)#otypes xmatched with skymapper through TOPCAT - 10'' radius
list(ssm)

allwise = pd.read_csv('/Users/Helios/Desktop/Research/CoolStars_Research/otypessmallwise2.csv', index_col=0)#otypes with skymapper xmatched with allwise through NOAO - 10'' radius
#list(allwise)

allwise_clean = allwise[allwise['cc_flags']=='0000']
allwise_clean.to_csv('/Users/Helios/Desktop/Research/CoolStars_Research/otypessmallwise_clean.csv')#cleaned data for NOAO

tmwise = pd.read_csv('/Users/Helios/Desktop/Research/CoolStars_Research/otypessmallwise2mass2.csv', index_col=0)#otypes with skymapper and wise xmatched with 2mass through NOAO - 10'' radius
list(tmwise)
Counter(tmwise['t1_t1_otype'])# 9519 LM*, 617 BD*, 809 other

gaia_matches = pd.read_csv('/Users/Helios/Desktop/Research/CoolStars_Research/gaia_sm_2mass_allwise_xmatch.csv',index_col=0) #These are the giants we're getting from Gaia DR 2
gaia_cuts = gaia_matches[(gaia_matches['flags'] == 0) & (gaia_matches['gal_contam'] == 0) & (gaia_matches['cc_flags'] == '0000') & (gaia_matches['cc_flg'] == '000') & (gaia_matches['ext_flg'] == 0) & (gaia_matches['w1snr'] >= 10) & (gaia_matches['w2snr'] >= 10) & (gaia_matches['j_snr'] >= 5) & (gaia_matches['k_snr'] >= 5) & (gaia_matches['h_snr'] >= 10) & (gaia_matches['w1nm'] > 8) & (gaia_matches['w2nm'] > 8)]
gaia_cuts.insert(loc=0, column='labels', value='other')

tmwise.columns = tmwise.columns.str.replace('t1_', '')#trimming the t1_ headers out
list(tmwise)
Counter(tmwise['otype'])

#indexing for the labels:
print(tmwise['otype'])

dwarfs = tmwise[tmwise['otype'].str.contains('low-mass')]
dwarfs.insert(loc=0, column='labels', value='lowmass*')

other = tmwise[~tmwise.main_id.isin(dwarfs.main_id)]#true false training set
other.insert(loc=0, column='labels', value='other')


train_set = [dwarfs,other]

main_tset = pd.concat(train_set)#labeled training set
Counter(main_tset['labels'])
list(main_tset)

mtset_cuts = main_tset[(main_tset['flags'] == 0) & (main_tset['cc_flags'] == 0) & (main_tset['cc_flg'] == '000') & (main_tset['ext_flg'] == 0) & (main_tset['w1snr'] >= 10) & (main_tset['w2snr'] >= 10) & (main_tset['j_snr'] >= 5) & (main_tset['k_snr'] >= 5) & (main_tset['h_snr'] >= 10) & (main_tset['w1nm'] > 8) & (main_tset['w2nm'] > 8)]
list(mtset_cuts)

Counter(mtset_cuts['sp_type'])

mtset2 = mtset_cuts[~mtset_cuts['sp_type'].str.contains('D|A|C|K|MIII|L9')]#cleaning...
Counter(mtset2['sp_type'])
#marking the absolute sptypes:
sps = mtset2['sp_type']

D1 = sps.replace([sps[sps.str.contains('M0')]], 'M0')
D2 = D1.replace([D1[D1.str.contains('M1')]], 'M1')
D3 = D2.replace([D2[D2.str.contains('M2')]], 'M2')
D4 = D3.replace([D3[D3.str.contains('M3')]], 'M3')
D5 = D4.replace([D4[D4.str.contains('M4')]], 'M4')
D6 = D5.replace([D5[D5.str.contains('M5')]], 'M5')
D7 = D6.replace([D6[D6.str.contains('M6')]], 'M6')
D8 = D7.replace([D7[D7.str.contains('M7')]], 'M7')
D9 = D8.replace([D8[D8.str.contains('M8')]], 'M8')
D10 = D9.replace([D9[D9.str.contains('M9')]], 'M9')

D11 = D10.replace([D10[D10.str.contains('L0')]], 'L0')
D12 = D11.replace([D11[D11.str.contains('L1')]], 'L1')
D13 = D12.replace([D12[D12.str.contains('L2')]], 'L2')
D14 = D13.replace([D13[D13.str.contains('L3')]], 'L3')
D15 = D14.replace([D14[D14.str.contains('L4')]], 'L4')
D16 = D15.replace([D15[D15.str.contains('L5')]], 'L5')
D17 = D16.replace([D16[D16.str.contains('L6')]], 'L6')
D18 = D17.replace([D17[D17.str.contains('L7')]], 'L7')
Counter(D18)
print(D18)

enum_dwarfs = D18

mtset2.insert(loc=0, column='abs_spt', value=enum_dwarfs)

Counter(mtset2['abs_spt'])

mtset3 = mtset2[mtset2.abs_spt != "b'M'"]#more cleaning....
Counter(mtset3['abs_spt'])

list(mtset3)
list(gaia_cuts)

#combining gaia and simbad tset:
gaia = gaia_cuts[['labels','object_id','raj2000', 'dej2000', 'u_psf', 'e_u_psf','g_psf','e_g_psf', 'r_psf', 'e_r_psf','i_psf', 'e_i_psf','z_psf','e_z_psf', 'j_m', 'j_cmsig','h_m', 'h_cmsig', 'k_m', 'k_cmsig', 'w1mpro', 'w1sigmpro','w2mpro','w2sigmpro']]


tset_n = mtset_cuts[['labels','object_id','raj2000', 'dej2000', 'u_psf', 'e_u_psf','g_psf','e_g_psf', 'r_psf', 'e_r_psf','i_psf', 'e_i_psf','z_psf','e_z_psf', 'j_m', 'j_cmsig','h_m', 'h_cmsig', 'k_m', 'k_cmsig', 'w1mpro', 'w1sigmpro','w2mpro','w2sigmpro']]
#We need to combine which columns are most important that these two datasets have in common. Otherwise, we can't combine the two datasets

mltraining1 = [gaia,tset_n]
mltraining1 = pd.concat(mltraining1)#machine learning training set without the skymapper data

mltraining1.to_csv('/Users/Helios/Desktop/Research/CoolStars_Research/mltraining1_noskymapper2.csv')
mtset3.to_csv('/Users/Helios/Desktop/Research/CoolStars_Research/otype_tset_tracer2.csv')#use this to reproduce the box plot and the other graphs as a sanity check
#tracer2 does not include brown dwarfs
Counter(mltraining1['labels'])

#from later time: cleaning phqual


