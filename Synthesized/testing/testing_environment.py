# -*- coding: utf-8 -*-
#
# Produced by: Synthesized Ltd
#
# The testing environment

from __future__ import division, print_function, absolute_import

import numpy as np
import plotly.figure_factory as ff
from plotly.offline import init_notebook_mode, iplot

__all__ = [
    'TestingEnvironment'
]


class TestingEnvironment:

    def compare_empirical_densities(self, realData, synthesizedData, targetField):

        init_notebook_mode()  # distribution of spending on the first month in the first category

        assert (len(synthesizedData[0]) > targetField),"Field value is out of range for the synthesized data"
        assert (len(realData[0]) > targetField),"Field value is out of range for the real data"

        fake_spendings = np.asarray([a[targetField] for a in synthesizedData])
        true_spendings = np.asarray([a[targetField] for a in realData])

        fake_spendings = np.asarray([a for a in fake_spendings if a != 0])
        true_spendings = np.asarray([a for a in true_spendings if a != 0])

        print("The sum of values in the field %s of synthesized data is %s" %(targetField, sum(fake_spendings)))
        print("The sum of values in the field %s of original data is %s" %(targetField, sum(true_spendings)))

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
        return(the_closest_fake_user, minimal_distance)

    def get_reconstruction_errors(self, original_dataset, original_dataset_batch, synthetic_dataset):
        from plotly.offline import init_notebook_mode, iplot

        init_notebook_mode()  # distribution of spending on the first month in the first category

        init_notebook_mode(connected=True)

        errors_synthetic = [self.get_the_closest_element(synthetic_dataset, original_dataset_batch[i])[1] for i in range(0,len(original_dataset_batch))]
        errors_original = [self.get_the_closest_element(original_dataset, original_dataset_batch[i])[1] for i in range(0,len(original_dataset_batch))]

        # Group data together
        # errors_scaled = [error/31 for error in errors]
        hist_data = [errors_original, errors_synthetic]

        group_labels = ['Errors for real', 'Errors for synthetic']

        # Create distplot with custom bin_size
        fig = ff.create_distplot(hist_data, group_labels, bin_size=[.001,.001])

        # Plot!
        iplot(fig, filename='jupyter/basic_bar')

#num = 54
#print(pd.DataFrame(data=np.asarray(testing_set[num]).reshape(32,4), index=np.arange(1,33), columns=np.arange(1,5)))

#errors = [get_the_closest_element(fake_data_set_as_arrays_of_arraya, testing_set_outlier[i])[1] for i in range(2,562,1)]
#errors[1:10]
#print(pd.DataFrame(data=np.asarray(synth).reshape(32,4), index=np.arange(1,33), columns=np.arange(1,5)))
