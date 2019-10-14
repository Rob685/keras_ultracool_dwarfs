import pandas as pd
import numpy as np
from collections import Counter
from xgboost import XGBClassifier
from xgboost import plot_importance
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from itertools import combinations
import seaborn as sns
import pickle
from datetime import datetime
import os


def XGBoost_Model(survey, path, train_data, feature_list, labels, test_size, n_iter, n_estimators):
    """
        Arguments:
            survey: could be either 'skymapper' or 'des' (dtype=str)
            path: path to where the results will be saved
            train_data: labeled data to train XGBoost.
            feature_list: list of magnitudes to train over.
            labels: list of labels from the train_data.
            test_size: fraction of the train data to train with leaving the rest
            for validation.
            n_iter: number of iterations to train xgboost with to choose the best model.

    """
    # Cleaning:
    # feature string needs to be defined for all magnitudes we train for
    tmags = train_data[feature_list + labels]
    tmags2 = tmags.dropna(how='any')
    tmags3 = tmags2[feature_list]

    # Combination function:
    def ccombinator(y):
        mags = pd.DataFrame(index=y.index)
        for a, b, in combinations(y.columns, 2):
            mags['{}-{}'.format(a, b)] = y[a] - y[b]
        c = mags
        return c

    train_colors = ccombinator(tmags3).join(tmags3, how='outer')
    ytarget = tmags2[labels]
    today = datetime.now()
    path = path
    directory = path + survey + today.strftime('%Y%m%d') + '/'
    os.makedirs(directory)

    accuracy_arr = []
    prediction_arr = []
    results_arr = []
    model_arr = []
    for i in range(n_iter):
        X_train, X_test, y_train, y_test = train_test_split(
            train_colors, ytarget, test_size=test_size)

        param = {'silet': False, 'scale_pos_weight': 0.5, 'eta': 0.5,
                 'objective': 'binary:logistic', 'n_estimators': n_estimators,
                 'max_delta_step': 5, 'max_depth': 5, 'tree_method': 'exact'}

        eval_set = [(X_train, y_train), (X_test, y_test)]

        model = XGBClassifier(silent=False,
                              scale_pos_weight=0.5,
                              eta=0.5,  # changed here to 0.5 from 0.1 to test (09/29)
                              colsample_bytree=0.8,  # tookout to test (09/29)
                              subsample=0.5,
                              objective='binary:logistic',
                              n_estimators=n_estimators,
                              alpha=0.5,  # took off to test (09/29)
                              max_delta_step=5,
                              max_depth=5,  # changed here to 5 from 10 to test (09/29)
                              gamma=10,  # took out to test (09/29)
                              tree_method='exact')  # changed to exact from approx to test (09/29ß)

        # modelXG.fit(X_train, y_train, eval_metric=["error", "rmse", "logloss"],
        # eval_set=eval_set, verbose=True)
        results = model.evals_result()

        # make predictions for test data
        predictions = modelXG.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print("Accuracy: %.10f%%" % (accuracy * 100.0))
        accuracy_arr.append(accuracy)
        prediction_arr.append(predictions)
        # evaluation_metrics_arr.append(eval_set)
        model_arr.append(modelXG)
        results_arr.append(results)
        print('Finished with step:', i)

    avg_accuracy = np.mean(accuracy_arr)
    sigma = np.std(accuracy_arr)

    print("Accuracy Standard Deviation: %.10f" % sigma)
    print('\n')
    print("Average Accuracy: %.10f%%" % (avg_accuracy*100))
    print('\n')
    print("Max Accuracy: %.10f%%" % (np.max(accuracy_arr)*100))

    plt.hist(accuracy_arr, histtype='step', bins=10, color='k')
    plt.xlabel('Accuracy Score')
    plt.ylabel('Frequency of Occurence')
    plt.title('Accuracy Distribution')
    plt.savefig(directory + 'accuracy_distribution.pdf')

    best_preds = prediction_arr[np.argmax(accuracy_arr)]

    best_model = model_arr[np.argmax(accuracy_arr)]
    pickle.dump(best_model, open(directory + "best_model.txt", "wb"))
    best_results = results_arr[np.argmax(accuracy_arr)]
    pickle.dump(best_results, open(directory + "best_results.txt", "wb"))
    all_results = [accuracy_arr, prediction_arr, results_arr, model_arr]
    pickle.dump(all_results, open(directory + "all_results.txt", "wb"))

    conmatrix = confusion_matrix(y_test, best_preds)
    print('Confusion Matrix:', conmatrix)
    print('Best Training predictions:', Counter(best_preds))

    plt.figure(figsize=(8, 8))
    plot_importance(best_model)
    plt.savefig(directory + 'feature_importance_plot.pdf')
    print('Length of training set:', len(train_colors))

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
    plt.savefig(directory + 'metric_plots.pdf')
    plt.show()
    return best_preds, best_model, best_results, all_results
