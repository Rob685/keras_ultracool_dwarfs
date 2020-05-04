import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.model_selection import train_test_split

def func(data, feature_list, label):
    tset = pd.read_csv(data)
    tmags = tset[feature_list + label]
    tmags2 = tmags.dropna(how='any')
    tmags3 = tmags2[feature_list]

    # Combination function:
    def ccombinator(x):
        mags = pd.DataFrame(index=x.index)
        for a, b, in combinations(x.columns, 2):
            mags['{}-{}'.format(a, b)] = x[a] - x[b]
        c = mags
        return c

    train_features = ccombinator(tmags3).join(tmags3, how='outer')

    ytarget = tmags2['label']

    binary_label = []
    for label in ytarget:
        if label == 'lowmass*':
             binary_label.append(1)
        else:
            binary_label.append(0)

    # print('Size of training features:',train_features.shape)
    train_colors = train_features.drop(feature_list,axis=1)
    # print('Size of training colors:',train_colors.shape)

    # splitting data into train and validation set:
    X_train, X_test, y_train, y_test = train_test_split(train_colors, binary_label,
                                                        test_size=0.20)

    #     np.shape(X_train),np.shape(y_train),np.shape(X_test),np.shape(y_test)
    return X_train,y_train,X_test,y_test
