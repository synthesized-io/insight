# -*- coding: utf-8 -*-
#
# Produced by: Synthesized Ltd
#
# The testing environment

from __future__ import division, print_function, absolute_import

import numpy as np
import pandas as pd
import plotly.figure_factory as ff
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split

from sklearn import metrics
from plotly.offline import init_notebook_mode, iplot

__all__ = [
    'TestingEnvironment'
]

# a good reference to logreg - https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8




def get_LogisticRegression_variables(dataset):
    X = ["operation", "amount"]
    y = ["type"]
    dataset = dataset[X + y].dropna()
    return {"X": dataset[X], "y": dataset[y]}


def get_stat_confidence(orig_dataset, synth_dataset, model=None):
    # X and y should be specified by a user
    orig_dataset = model["variables"](orig_dataset)
    synth_dataset = model["variables"](synth_dataset)

    orig_dataset_X, orig_dataset_y = orig_dataset["X"], orig_dataset["y"]
    synth_dataset_X, synth_dataset_y = synth_dataset["X"], synth_dataset["y"]

    dataset1_X_train, dataset1_X_test, dataset1_y_train, dataset1_y_test = train_test_split(orig_dataset_X,
                                                                                            orig_dataset_y,
                                                                                            test_size=0.2,
                                                                                            random_state=0)
    dataset2_X_train, dataset2_X_test, dataset2_y_train, dataset2_y_test = train_test_split(synth_dataset_X,
                                                                                            synth_dataset_y,
                                                                                            test_size=0.2,
                                                                                            random_state=0)
    model_synth = model["model"]()
    model_oracle = model["model"]()
    model_synth.fit(dataset2_X_train, dataset2_y_train.values.ravel())
    model_oracle.fit(dataset1_X_train, dataset1_y_train.values.ravel())

    # y_pred_oracle = model_oracle.predict(dataset1_X_test)
    # y_pred_synthetic = model_synth.predict(dataset1_X_test)

    oracle_score, synth_score = model_oracle.score(dataset1_X_test, dataset1_y_test), model_synth.score(dataset1_X_test,
                                                                                                        dataset1_y_test)
    return oracle_score, synth_score, abs(1 - abs(oracle_score - synth_score) / oracle_score)


synth_stat_models = [{"model": LogisticRegression, "variables": get_LogisticRegression_variables}]
#               {"model": LinearRegression, "variables": self.get_LinearRegression_variables}]

def compare_pred_performance(dataset1, dataset2, user_stat_models=[]):
    for stat_model in synth_stat_models:
        print(
            "Prediction confidence for %s when trained on synthetic data: %.3f \nOracle score : %.3f \nSynth score : %.3f" %
            (stat_model["model"].__name__, get_stat_confidence(dataset1, dataset2, stat_model)[2],
             get_stat_confidence(dataset1, dataset2, stat_model)[0],
             get_stat_confidence(dataset1, dataset2, stat_model)[1]))

    for stat_model in user_stat_models:
        get_stat_confidence(dataset1, dataset2, stat_model)


class TestingEnvironment:
    def __init__(self):
        self.synth_stat_models = [{"model" : LogisticRegression, "variables" : self.get_LogisticRegression_variables},
                                  {"model": LinearRegression, "variables": self.get_LinearRegression_variables}]

    def get_LogisticRegression_variables(self, dataset):
        X = ["operation", "amount", "balance"]
        y = ["type"]
        dataset = dataset[X + y].dropna()
        return {"X" : dataset[X], "y" : dataset[y]}

    def get_LinearRegression_variables(self, dataset):
        dataset['date'] = pd.to_datetime(dataset['date'])
        account_ids = dataset["account_id"].unique()
        operations = dataset["operation"].unique()
        X = np.array([]).reshape(0,5)
        y = np.array([])
        for account_id in account_ids:
            user = dataset[dataset["account_id"] == account_id]
            for month in range(1,12):
                X_local = np.array([])
                user_local = user[ user["date"].dt.month == month]
                for operation in operations:
                    X_local = np.append(X_local, user_local[user_local["operation"] == operation]["amount"].sum())
                X = np.vstack((X, X_local))
                next_month_df = user[(user["date"].dt.month == month + 1) & (user["operation"] == 2)]
                y = np.concatenate((y, np.array([next_month_df["amount"].sum() ] )))
        return {"X" : X, "y" : y}

    def get_stat_confidence(self, original_dataset, synthetic_dataset, model = None):
        #X and y should be specified by a user
        original_dataset = model["variables"](original_dataset)
        synthetic_dataset = model["variables"](synthetic_dataset)

        original_dataset_X, original_dataset_y  = original_dataset["X"], original_dataset["y"]
        synthetic_dataset_X, synthetic_dataset_y = synthetic_dataset["X"], synthetic_dataset["y"]

        dataset1_X_train, dataset1_X_test, dataset1_y_train, dataset1_y_test = train_test_split(original_dataset_X, original_dataset_y, test_size=0.2, random_state=0)
        dataset2_X_train, dataset2_X_test, dataset2_y_train, dataset2_y_test = train_test_split(synthetic_dataset_X, synthetic_dataset_y, test_size=0.2, random_state=0)
        model_synth = model["model"]()
        model_oracle = model["model"]()
        model_synth.fit(dataset2_X_train, dataset2_y_train)
        model_oracle.fit(dataset1_X_train, dataset1_y_train)

       # y_pred_oracle = model_oracle.predict(dataset1_X_test)
       # y_pred_synthetic = model_synth.predict(dataset1_X_test)

        oracle_score, synth_score = model_oracle.score(dataset1_X_test, dataset1_y_test), model_synth.score(dataset1_X_test, dataset1_y_test)
       # print(oracle_score, synth_score, 1 - abs(oracle_score - synth_score)/ synth_score)
        return(1 - abs(oracle_score - synth_score) / synth_score)


    def compare_pred_performance(self, dataset1, dataset2, user_stat_models = []):
        for stat_model in self.synth_stat_models:
            print("Prediction confidence for %s when trained on synthetic data: %.3f" %
                  (stat_model["model"].__name__, self.get_stat_confidence(dataset1, dataset2, stat_model)) )

        for stat_model in user_stat_models:
            self.get_stat_confidence(dataset1, dataset2, stat_model)

    def compare_empirical_densities(self, original_dataset, synthetic_dataset, target_field):

        init_notebook_mode()  # distribution of spending on the first month in the first category

        assert (len(synthetic_dataset[0]) > target_field), "Field value is out of range for the synthesized data"
        assert (len(original_dataset[0]) > target_field), "Field value is out of range for the real data"

        fake_spendings = np.asarray([a[target_field] for a in synthetic_dataset])
        true_spendings = np.asarray([a[target_field] for a in original_dataset])

        fake_spendings = np.asarray([a for a in fake_spendings if a != 0])
        true_spendings = np.asarray([a for a in true_spendings if a != 0])

        print("The sum of values in the field %s of synthesized data is %s" % (target_field, sum(fake_spendings)))
        print("The sum of values in the field %s of original data is %s" % (target_field, sum(true_spendings)))

        assert (sum(fake_spendings) > 0), "zero value in the field in the synthesized dataset"
        assert (sum(true_spendings) > 0), "zero value in the field in the read dataset"

        init_notebook_mode(connected=True)
        # Group data together
        hist_data = [true_spendings, fake_spendings]

        group_labels = ['Real data', 'Synthetic data']

        # Create distplot with custom bin_size
        fig = ff.create_distplot(hist_data, group_labels, bin_size=[.01, .01])

        # Plot!
        iplot(fig, filename='jupyter/basic_bar')

    def get_the_closest_element(self, data_set, element):
        minimal_distance = 100
        the_closest_fake_user = 0
        for user in data_set:
            if np.sum(abs(element - user)) < minimal_distance:
                the_closest_fake_user = user
                minimal_distance = np.sum(abs(element - user))
        return (the_closest_fake_user, minimal_distance)

    def get_reconstruction_errors(self, original_dataset, original_dataset_batch, synthetic_dataset):
        from plotly.offline import init_notebook_mode, iplot

        init_notebook_mode()  # distribution of spending on the first month in the first category

        init_notebook_mode(connected=True)

        errors_synthetic = [self.get_the_closest_element(synthetic_dataset, original_dataset_batch[i])[1] for i in
                            range(0, len(original_dataset_batch))]
        errors_original = [self.get_the_closest_element(original_dataset, original_dataset_batch[i])[1] for i in
                           range(0, len(original_dataset_batch))]

        # Group data together
        # errors_scaled = [error/31 for error in errors]
        hist_data = [errors_original, errors_synthetic]

        group_labels = ['Errors for real', 'Errors for synthetic']

        # Create distplot with custom bin_size
        fig = ff.create_distplot(hist_data, group_labels, bin_size=[.001, .001])

        # Plot!
        iplot(fig, filename='jupyter/basic_bar')

# num = 54
# print(pd.DataFrame(data=np.asarray(testing_set[num]).reshape(32,4), index=np.arange(1,33), columns=np.arange(1,5)))

# errors = [get_the_closest_element(fake_data_set_as_arrays_of_arraya, testing_set_outlier[i])[1] for i in range(2,562,1)]
# errors[1:10]
# print(pd.DataFrame(data=np.asarray(synth).reshape(32,4), index=np.arange(1,33), columns=np.arange(1,5)))
