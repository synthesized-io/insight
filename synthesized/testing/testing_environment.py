# -*- coding: utf-8 -*-
#
# Produced by: Synthesized Ltd
#
# The testing environment

from __future__ import division, print_function, absolute_import

import numpy as np
import plotly.figure_factory as ff
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cross_validation import train_test_split

from sklearn import metrics
from plotly.offline import init_notebook_mode, iplot

__all__ = [
    'TestingEnvironment'
]

# a good reference to logreg - https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8

class TestingEnvironment:
    def __init__(self):
        self.synth_stat_models = [{"model" : LogisticRegression, "X" : ["operation", "amount", "balance"], "y" : ["type"]}]

    def get_stat_confidence(self, original_dataset, synthetic_dataset, model = None):
        #X and y should be specified by a user
        X = model["X"]
        y = model["y"]
        original_dataset = original_dataset[X + y].dropna()
        synthetic_dataset = synthetic_dataset[X + y].dropna()

        dataset1_X, dataset1_y  = original_dataset[X], original_dataset[y]
        dataset2_X, dataset2_y = synthetic_dataset[X], synthetic_dataset[y]

        dataset1_X_train, dataset1_X_test, dataset1_y_train, dataset1_y_test = train_test_split(dataset1_X, dataset1_y, test_size=0.2, random_state=0)
        dataset2_X_train, dataset2_X_test, dataset2_y_train, dataset2_y_test = train_test_split(dataset2_X, dataset2_y, test_size=0.2, random_state=0)
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
