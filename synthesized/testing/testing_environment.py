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

from sklearn import clone
from plotly.offline import init_notebook_mode, iplot
from sklearn.preprocessing import StandardScaler


def to_categorical(a, edges):
    def to_bucket(x):
        idx = np.searchsorted(edges, x)
        if idx == len(edges) - 1:
            idx -= 1
        start = edges[idx]
        end = edges[idx + 1]
        return "[{}-{})".format(start, end)

    to_bucketv = np.vectorize(to_bucket)
    return to_bucketv(a)

# a good reference to logreg - https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8


def estimate_utility(df_orig, df_synth, categorical_columns, continuous_columns, classifier=LogisticRegression(), min_score=0.01):
    categorical_columns_set = set(categorical_columns)
    continuous_columns_set = set(continuous_columns)
    intersection = continuous_columns_set.intersection(categorical_columns_set)
    if len(intersection) > 0:
        raise ValueError('Columns should be either continuous or categorical: {}'.format(intersection))
    columns = continuous_columns_set.union(categorical_columns_set)
    df_orig = df_orig.apply(pd.to_numeric)
    df_synth = df_synth.apply(pd.to_numeric)
    result = []
    for column in columns:
        X_columns = list(columns.difference([column]))

        X_orig = df_orig[X_columns]
        y_orig = df_orig[column]

        X_synth = df_synth[X_columns]
        y_synth = df_synth[column]

        if column in continuous_columns_set:
            _, edges = np.histogram(y_orig)
            y_orig = to_categorical(y_orig, edges)
            y_synth = to_categorical(y_synth, edges)

        X_orig = StandardScaler().fit_transform(X_orig)
        X_synth = StandardScaler().fit_transform(X_synth)

        X_orig_train, X_orig_test, y_orig_train, y_orig_test = train_test_split(X_orig, y_orig, test_size=0.2, random_state=0)
        X_synth_train, X_synth_test, y_synth_train, y_synth_test = train_test_split(X_synth, y_synth, test_size=0.2, random_state=0)

        baseline_score = clone(classifier).fit(X_orig_train, y_orig_train).score(X_orig_test, y_orig_test)
        synth_score = clone(classifier).fit(X_synth_train, y_synth_train).score(X_orig_test, y_orig_test)
        if baseline_score < min_score or synth_score < min_score:
            utility = float('nan')
        else:
            diff = baseline_score - synth_score
            utility = round((1 - diff / baseline_score) * 100, 2)
        result.append({
            'target_column': column,
            'classifier': classifier.__class__.__name__,
            'baseline_score': baseline_score,
            'synth_score': synth_score,
            'change': synth_score - baseline_score,
            'utility': utility,
        })

    return pd.DataFrame.from_records(result, columns=['target_column', 'classifier', 'baseline_score', 'synth_score', 'change', 'utility'])


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
