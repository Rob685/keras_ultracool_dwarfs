import pandas as pd
import numpy as np
from collections import Counter
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from itertools import combinations
import seaborn as sns


def XGBoost_Model(train_data, test_size, n_iter):
    # Cleaning:
    tmags = train_data[['i_psf', 'z_psf',
                        'h_m', 'j_m', 'k_m',
                        'w1mpro', 'w2mpro', 'label']]  # Dataset NEEDS to have these column names for this function to work.
    tmags2 = tmags.dropna(how='any')
    tmags3 = tmags2[['i_psf', 'z_psf',
                     'h_m', 'j_m', 'k_m',
                     'w1mpro', 'w2mpro']]

    # Combination function:
    def ccombinator(y):
        mags = pd.DataFrame(index=y.index)
        for a, b, in combinations(y.columns, 2):
            mags['{}-{}'.format(a, b)] = y[a] - y[b]
        c = mags
        return c

    train_colors = ccombinator(tmags3).join(tmags3, how='outer')
    ytarget = tmags2['label']

    accuracy_arr = []
    prediction_arr = []
    results_arr = []
    model_arr = []
    for i in range(n_iter):
        test_size = 0.50
        X_train, X_test, y_train, y_test = train_test_split(
            train_colors, ytarget, test_size=test_size)

        model = XGBClassifier(silent=False,
                              scale_pos_weight=0.5,
                              eta=0.1,
                              colsample_bytree=0.8,
                              subsample=0.5,
                              objective='binary:logistic',
                              n_estimators=100,
                              reg_alpha=0.5,
                              max_delta_step=5,
                              max_depth=10,
                              gamma=10,
                              tree_method='approx')

        eval_set = [(X_train, y_train), (X_test, y_test)]
        model.fit(X_train, y_train, eval_metric=["error", "rmse", "logloss"],
                  eval_set=eval_set, verbose=True)
        results = model.evals_result()

        # make predictions for test data
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print("Accuracy: %.10f%%" % (accuracy * 100.0))
        accuracy_arr.append(accuracy)
        prediction_arr.append(predictions)
        # evaluation_metrics_arr.append(eval_set)
        model_arr.append(model)
        results_arr.append(results)
        print('Finished with step:', i)

    avg_accuracy = np.mean(accuracy_arr)
    sigma = np.std(accuracy_arr)

    print("Accuracy Standard Deviation: %.10f" % sigma)
    print('\n')
    print("Average Accuracy: %.10f%%" % (avg_accuracy*100))
    print('\n')
    print("Max Accuracy:", np.max(accuracy_arr)*100)

    plt.hist(accuracy_arr, histtype='step', bins='sqrt', color='k')
    plt.xlabel('Accuracy Score')
    plt.ylabel('Frequency of Occurence')
    plt.title('Accuracy Distribution')
    plt.savefig('/Users/roberttejada/coolstarsucsd/accuracy_distribution_skymapper_training.pdf')

    best_preds = prediction_arr[np.argmax(accuracy_arr)]
    best_model = model_arr[np.argmax(accuracy_arr)]
    best_results = results_arr[np.argmax(accuracy_arr)]

    conmatrix = confusion_matrix(y_test, best_preds)
    print('Confusion Matrix:', conmatrix)
    print('Best Training predictions:', Counter(best_preds))

    return best_preds, best_model, best_results
