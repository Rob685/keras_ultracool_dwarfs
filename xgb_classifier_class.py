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
import pickle
from datetime import datetime
import os


def ccombinator(y):
    mags = pd.DataFrame(index=y.index)
    for a, b, in combinations(y.columns, 2):
        mags['{}-{}'.format(a, b)] = y[a] - y[b]
    c = mags
    return c

class XGBoost_Class:
    def __init__(self,survey, train_data, feature_list, labels, test_size, n_iter,n_estimators):
        self.survey = survey
        self.train_data = train_data
        self.feature_list = feature_list
        self.labels = labels
        self.test_size = test_size
        self.n_iter = n_iter
        self.n_estimators = n_estimators

    def xgboost_classifier(self):
        """
            Arguments:
                survey: could be either 'skymapper' or 'des' (dtype=str)
                train_data: labeled data to train XGBoost.
                feature_list: list of magnitudes to train over.
                labels: list of labels from the train_data.
                test_size: fraction of the train data to train with leaving the rest
                for validation.
                n_iter: number of iterations to train xgboost with to choose the best model.

        """
        # Cleaning:
        # feature string needs to be defined for all magnitudes we train for
        tmags = self.train_data[self.feature_list + self.labels]
        tmags2 = tmags.dropna(how='any')
        tmags3 = tmags2[self.feature_list]

        train_colors = ccombinator(tmags3).join(tmags3, how='outer')
        ytarget = tmags2[labels]
        today = datetime.now()
        path = '/Users/roberttejada/Desktop/'
        self.directory = path + survey + today.strftime('%Y%m%d') + '/'
        os.makedirs(self.directory)

        self.accuracy_arr = []
        self.prediction_arr = []
        self.results_arr = []
        self.model_arr = []
        for i in range(self.n_iter):
            X_train, X_test, y_train, y_test = train_test_split(
                train_colors, ytarget, test_size=self.test_size)

            model = XGBClassifier(silent=False,
                                  scale_pos_weight=0.5,
                                  eta=0.5, #changed here to 0.5 from 0.1 to test (09/29)
                                  #colsample_bytree=0.8, tookout to test (09/29)
                                  subsample=0.5,
                                  objective='binary:logistic',
                                  n_estimators=self.n_estimators,
                                  #alpha=0.5, took off to test (09/29)
                                  max_delta_step=5,
                                  max_depth=5, #changed here to 5 from 10 to test (09/29)
                                  #gamma=10, took out to test (09/29)
                                  tree_method='exact') #changed to exact from approx to test (09/29ß)

            eval_set = [(X_train, y_train), (X_test, y_test)]
            model.fit(X_train, y_train, eval_metric=["error", "rmse", "logloss"],
                      eval_set=eval_set, verbose=True)
            results = model.evals_result()

            # make predictions for test data
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            print("Accuracy: %.10f%%" % (accuracy * 100.0))
            self.accuracy_arr.append(accuracy)
            self.prediction_arr.append(predictions)
            # evaluation_metrics_arr.append(eval_set)
            self.model_arr.append(model)
            self.results_arr.append(results)
            print('Finished with step:', i)

        avg_accuracy = np.mean(accuracy_arr)
        sigma = np.std(accuracy_arr)

        print("Accuracy Standard Deviation: %.10f" % sigma)
        print('\n')
        print("Average Accuracy: %.10f%%" % (avg_accuracy*100))
        print('\n')
        print("Max Accuracy: %.10f%%" % (np.max(accuracy_arr)*100))
        self.best_preds = self.prediction_arr[np.argmax(self.accuracy_arr)]

        self.best_model = self.model_arr[np.argmax(self.accuracy_arr)]
        pickle.dump(self.best_model, open(self.directory + "best_model.txt", "wb"))
        self.best_results = self.results_arr[np.argmax(self.accuracy_arr)]
        pickle.dump(self.best_results, open(self.directory + "best_results.txt", "wb"))
        self.all_results = [self.accuracy_arr, self.prediction_arr, self.results_arr, self.model_arr]
        pickle.dump(self.all_results, open(self.directory + "all_results.txt", "wb"))

        conmatrix = confusion_matrix(y_test, self.best_preds)
        print('Confusion Matrix:', conmatrix)
        print('Best Training predictions:', Counter(self.best_preds))

        plot_importance(self.best_model)
        plt.savefig(self.directory + 'feature_importance_plot.pdf')
        print('Length of training set:', len(train_colors))

        self.results = self.best_model.evals_result()

        return self.best_preds, self.best_model, self.best_results, self.all_results

    def accuracy_histogram(self):
        plt.hist(self.accuracy_arr, histtype='step', bins=10, color='k')
        plt.xlabel('Accuracy Score')
        plt.title('Accuracy Distribution')
        plt.savefig(self.directory + 'accuracy_distribution_training.pdf')

    def plot_metrics(self):
        epochs = range(len(self.best_results['validation_0']['error']))
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
