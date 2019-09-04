import xgb_general_argmax_func as xgbooster
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
from sklearn.model_selection import train_test_split
from collections import Counter
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.colors import LogNorm
import pickle

train = pd.read_csv('/Users/roberttejada/Desktop/des_gaia_data_ml/des_training_set.csv')


def abs_mag(p, m):
    return m - 5*np.log10((1000/(p)).astype(np.float64)) + 5


def ccombinator(y):
    mags = pd.DataFrame(index=y.index)
    for a, b, in combinations(y.columns, 2):
        mags['{}-{}'.format(a, b)] = y[a] - y[b]
    c = mags
    return c


flist = ['MAG_AUTO_I', 'MAG_AUTO_Z', 'h_m', 'j_m', 'k_m', 'w1mpro', 'w2mpro']
label_list = ['labels']

best_preds, best_model, best_results, all_results_skymapper = xgbooster.XGBoost_Model(
    'DES', train, flist, label_list, 0.20, 100)

results = best_model.evals_result()
epochs = range(len(best_results['validation_0']['error']))
# plot log loss
fig, ax = plt.subplots(3, 1, figsize=(10, 10))
ax[0].plot(epochs, results['validation_0']['logloss'], 'k--', label='Train')
ax[0].plot(epochs, results['validation_1']['logloss'], label='Validation', lw=6, alpha=0.5)
ax[0].legend()
ax[0].set_ylabel('Log Loss')

ax[1].plot(epochs, results['validation_0']['error'], 'k--', label='Train')
ax[1].plot(epochs, results['validation_1']['error'], label='Validation', lw=5, alpha=0.5)
ax[1].legend()
ax[1].set_ylabel('Classification Error')

ax[2].plot(epochs, results['validation_0']['rmse'], 'k--', label='Train')
ax[2].plot(epochs, results['validation_1']['rmse'], label='Validation', lw=5, alpha=0.5)
ax[2].legend()
ax[2].set_ylabel('RMS Error')

plt.tight_layout()
plt.savefig('/Users/roberttejada/coolstarsucsd/xgb_metric_plots_des_80train.pdf')


target_refset = pd.read_csv(
    '/Users/roberttejada/Desktop/des_gaia_data_ml/des_refset_gaia_allwise_twomass_sdss12.csv')

target_refset['M_G'] = abs_mag(target_refset['parallax'].values,
                               target_refset['phot_g_mean_mag'].values)

refset = target_refset[['COADD_OBJECT_ID', 'MAG_AUTO_I', 'MAG_AUTO_Z',
                        'Hmag', 'Jmag', 'Kmag', 'W1mag', 'W2mag'
                        # ,'M_G'
                        ]].dropna(how='any')

refset_4preds = refset[['MAG_AUTO_I', 'MAG_AUTO_Z',
                        'Hmag', 'Jmag', 'Kmag', 'W1mag', 'W2mag'
                        # ,'M_G'
                        ]]

refset_4preds = refset_4preds.rename(index=str, columns={"Hmag": "h_m",
                                                         "Jmag": "j_m", "Kmag": "k_m", "W1mag": "w1mpro", "W2mag": "w2mpro"})
print('refset read')


# In[10]:


features_colors = ccombinator(refset_4preds)

features_ref = features_colors.join(refset_4preds, how='outer')

despredictions = best_model.predict(features_ref)
print('SkyMapper Predictions:', Counter(despredictions))


# In[ ]:


refset.insert(loc=8, column='xgb_predictions', value=smpredictions)

df = refset[['object_id', 'xgb_predictions']]

refset_wpreds = target_refset.merge(df, how='inner', on='object_id')

# In[ ]:


g_rp = refset_wpreds['phot_g_mean_mag'] - refset_wpreds['phot_rp_mean_mag']

refset_wpreds['g_rp'] = g_rp
refset_wpreds.to_csv(
    '/Users/roberttejada/Desktop/des_gaia_data_ml/des_refset_wpredictions_80train.csv')

# In[ ]:


refset_par = refset_wpreds[(refset_wpreds['parallax'] > 0)]


# In[ ]:


gaia_test = refset_par[['g_rp', 'phot_g_mean_mag', 'parallax', 'M_G',
                        'xgb_predictions']].dropna(how='any')

giants_pred = gaia_test[gaia_test['xgb_predictions'] == 'giant']
dwarfs_pred = gaia_test[gaia_test['xgb_predictions'] == 'dwarf']


# In[ ]:


print('giant/dwarf ratio is:', len(giants_pred)/len(dwarfs_pred))
print('Giant predictions:', len(giants_pred))
print('Dwarf predictions:', len(dwarfs_pred))


sns.set_style("white")

plt.figure(figsize=(10, 8))
plt.scatter(dwarfs_pred['g_rp'],
            abs_mag(dwarfs_pred['parallax'].values,
                    dwarfs_pred['phot_g_mean_mag'].values), s=1, c='k', alpha=0.05,
            label='pred. dwarfs')
plt.scatter(giants_pred['g_rp'],
            abs_mag(giants_pred['parallax'].values,
                    giants_pred['phot_g_mean_mag'].values), s=1, c='b', alpha=0.05,
            label='pred. other')
plt.axhline(y=5, ls='--', c='k')
plt.xlabel(r'$G - RP$')
plt.ylabel(r'$M_G$ (absolute mag)')
plt.xlim(0.0, 2.5)
# plt.legend()
# plt.tight_layout()
plt.gca().invert_yaxis()
plt.title('SkyMapper Predictions: XGBoost')

k = sns.color_palette("Greys")[2]
b = sns.color_palette("Blues")[2]
#g = sns.color_palette("Greens")[2]

black_patch = mpatches.Patch(color=k, label='dwarfs')
blue_patch = mpatches.Patch(color=b, label='giants')
#green_patch = mpatches.Patch(color=g, label='LIC')

sns.reset_orig
plt.legend(handles=[black_patch, blue_patch])
plt.minorticks_on()
plt.savefig('/Users/roberttejada/coolstarsucsd/des_xgb_predictions_gaiaplot_80train.pdf')
