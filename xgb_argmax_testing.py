import xgb_argmax_func as xgbooster
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

train = pd.read_csv('/Users/roberttejada/Desktop/gaia_data_ml/all_gaiasmdata_readyforml.csv')


def abs_mag(p, m):
    return m - 5*np.log10((1000/(p)).astype(np.float64)) + 5


def ccombinator(y):
    mags = pd.DataFrame(index=y.index)
    for a, b, in combinations(y.columns, 2):
        mags['{}-{}'.format(a, b)] = y[a] - y[b]
    c = mags
    return c


train['M_G'] = abs_mag(train['parallax'].values,
                       train['phot_g_mean_mag'].values)

best_preds, best_model, best_results = xgbooster.XGBoost_Model(train, 0.50, 50)

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
plt.savefig('/Users/roberttejada/coolstarsucsd/xgb_metric_plots_skymapper_100iters.pdf')


skymapper_refset = pd.read_csv('/Users/roberttejada/Desktop/gaia_data_ml/skymapper_merged_all.csv')

skymapper_refset['M_G'] = abs_mag(skymapper_refset['parallax'].values,
                                  skymapper_refset['phot_g_mean_mag'].values)

refset = skymapper_refset[['object_id', 'i_psf', 'z_psf',
                           'Hmag_x', 'Jmag_x', 'Kmag_x', 'W1mag', 'W2mag'
                           # ,'M_G'
                           ]].dropna(how='any')

refset_4preds = refset[['i_psf', 'z_psf',
                        'Hmag_x', 'Jmag_x', 'Kmag_x', 'W1mag', 'W2mag'
                        # ,'M_G'
                        ]]

refset_4preds = refset_4preds.rename(index=str, columns={"Hmag_x": "h_m",
                                                         "Jmag_x": "j_m", "Kmag_x": "k_m", "W1mag": "w1mpro", "W2mag": "w2mpro"})
print('refset read')


# In[10]:


features_colors = ccombinator(refset_4preds)

features_ref = features_colors.join(refset_4preds, how='outer')

smpredictions = best_model.predict(features_ref)
print('SkyMapper Predictions:', Counter(smpredictions))


# In[ ]:


refset.insert(loc=8, column='xgb_predictions', value=smpredictions)

df = refset[['object_id', 'xgb_predictions']]

smrefset_wpreds = skymapper_refset.merge(df, how='inner', on='object_id')


# In[ ]:


g_rp = smrefset_wpreds['phot_g_mean_mag'] - smrefset_wpreds['phot_rp_mean_mag']

smrefset_wpreds['g_rp'] = g_rp


# In[ ]:


smrefset_par = smrefset_wpreds[(smrefset_wpreds['parallax'] > 0)]


# In[ ]:


gaia_test = smrefset_par[['g_rp', 'phot_g_mean_mag', 'parallax', 'M_G',
                          'xgb_predictions']].dropna(how='any')

giants_pred = gaia_test[gaia_test['xgb_predictions'] == 'other']
dwarfs_pred = gaia_test[gaia_test['xgb_predictions'] == 'lowmass*']


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
plt.savefig('/Users/roberttejada/coolstarsucsd/skymapper_xgb_predictions_gaiaplot.pdf')
