import pandas as pd
import numpy as np
from collections import Counter
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib

matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from itertools import combinations
import seaborn as sns
import matplotlib.patches as mpatches

train = pd.read_csv('/Users/roberttejada/Desktop/gaia_data_ml/all_gaiasmdata_readyforml.csv')  # Training labeled set

# Cleaning:
tmags = train[['i_psf', 'z_psf', 'h_m', 'j_m', 'k_m', 'w1mpro', 'w2mpro',
               'label']]
# Counter(train['StarType'])
tmags2 = tmags.dropna(how='any')
# Counter(tmags2['StarType'])
tmags3 = tmags2[['i_psf', 'z_psf', 'h_m', 'j_m', 'k_m', 'w1mpro', 'w2mpro']]

# Mixing the magnitudes to form the colors:


def ccombinator(y):
    mags1 = pd.DataFrame(index=y.index)
    mags2 = pd.DataFrame(index=y.index)
    for a, b, in combinations(y.columns, 2):
        mags1['{}-{}'.format(a, b)] = y[a] - y[b]
        mags2['{}-{}'.format(b, a)] = y[b] - y[a]
    c = mags1.join(mags2)
    return c


# Joining the color and magnitudes
features = ccombinator(tmags3).join(tmags3, how='outer')

labels = tmags2.label

test_size = 0.20
X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                                    test_size=test_size)

#----------------------fit model no training data------------------------------#

model = XGBClassifier()
model.fit(X_train, y_train)
print('Model Fitted')

# make predictions for test data
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.10f%%" % (accuracy * 100.0))

# Predictions:
conmatrix = confusion_matrix(y_test, predictions)
print('Confusion Matrix:', conmatrix)
print('Training predictions:', Counter(predictions))

#------------------------SkyMapper Predictions---------------------------------#
print('Initizalizing SkyMapper Predictions')
skymapper_refset = pd.read_csv('/Users/roberttejada/Desktop/gaia_data_ml/skymapper_merged_all.csv')
print('skymapper reference read')

refset = skymapper_refset[['object_id', 'i_psf', 'z_psf',
                           'Hmag_x', 'Jmag_x', 'Kmag_x', 'W1mag', 'W2mag']].dropna(how='any')

refset_4preds = refset[['i_psf', 'z_psf',
                        'Hmag_x', 'Jmag_x', 'Kmag_x', 'W1mag', 'W2mag']]

refset_4preds = refset_4preds.rename(index=str, columns={"Hmag_x": "h_m",
                                                         "Jmag_x": "j_m", "Kmag_x": "k_m", "W1mag": "w1mpro", "W2mag": "w2mpro"})
print('refset read')
features_colors = ccombinator(refset_4preds)

features_ref = features_colors.join(refset_4preds, how='outer')

smpredictions = model.predict(features_ref)
print('SkyMapper Predictions:', Counter(smpredictions))

#---------------------------Gaia Analysis--------------------------------------#

# Absolute Magntude equation:
print('Initizalizing Gaia Analysis')

def abs_mag(p, m):
    return m - 5*np.log10((1000/(p)).astype(np.float64)) + 5


refset.insert(loc=8, column='xgb_predictions', value=smpredictions)

df = refset[['object_id', 'xgb_predictions']]

smrefset_wpreds = skymapper_refset.merge(df, how='inner', on='object_id')

# Getting the G-RP color for magnitude analysis:
g_rp = smrefset_wpreds['phot_g_mean_mag'] - smrefset_wpreds['phot_rp_mean_mag']

smrefset_wpreds.insert(loc=195, column='g_rp', value=g_rp)

# No negative parallaxes:
smrefset_par = smrefset_wpreds[(smrefset_wpreds['parallax'] >= 0)]

gaia_test = smrefset_par[['g_rp', 'phot_g_mean_mag', 'parallax',
                          'xgb_predictions']].dropna(how='any')

giants_pred = gaia_test[gaia_test['xgb_predictions'] == 'other']
dwarfs_pred = gaia_test[gaia_test['xgb_predictions'] == 'lowmass*']

#----------------------------Plot Results--------------------------------------#
sns.set_style("white")

plt.scatter(dwarfs_pred['g_rp'],
            abs_mag(dwarfs_pred['parallax'].values,
                    dwarfs_pred['phot_g_mean_mag'].values), s=1, c='k', alpha=0.1,
            label='pred. dwarfs')
plt.scatter(giants_pred['g_rp'],
            abs_mag(giants_pred['parallax'].values,
                    giants_pred['phot_g_mean_mag'].values), s=1, c='b', alpha=0.1,
            label='pred. other')
plt.xlabel(r'$G - RP$')
plt.ylabel(r'$M_G$ (absolute mag)')
plt.xlim(-0.5, 2.0)
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

plt.legend(handles=[black_patch, blue_patch])
#plt.title('SkyMapper Training Set')
plt.show()
print('XGB Script Finished!')
