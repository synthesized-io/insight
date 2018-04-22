# -*- coding: utf-8 -*-
#
# Produced by: Synthesized Ltd
#
# Test the AlphaSynth

from __future__ import division, absolute_import, division

import os

import numpy as np

from synthesized.modules.data_analytics_tools import DataPipeline
from synthesized.modules.synth import AlphaSynth

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split

TESTDATA_FILENAME = os.path.join(os.path.dirname(__file__), '../../data/transactions_sample_10k.csv')


def test_AlphaSynth():
    pipeline = DataPipeline()
    pipeline.load_from_csv(TESTDATA_FILENAME)
    # targetDataSe
    feature_list = ["account_id", "date", "operation", "amount"]
    pipeline.preprocess_crude_dataset(feature_list)

    # targetDataSe
    targetDataSet = pipeline.working_dataset
    targetDataSet.columns = ['seqId', 'date', 'category', 'value']

    pipeline.prepare_dataset_for_training()

    training_set = pipeline.training_dataset

    all_data = np.asarray(training_set)

    seed = 42
    X_train, X_test = train_test_split(all_data, train_size=0.7, random_state=seed)

    # define
    alpha = AlphaSynth(n_epochs=100, n_hidden=250, learning_rate=0.01, batch_size=106,
                       display_step=10, activation_function='relu', verbose=2, min_change=1e-6,
                       random_state=seed, clip=True, l2_penalty=1e-5,
                       early_stopping=True)

    # fit
    alpha.fit(X_train)

    reconstructed = alpha.feed_forward(X_test)

    # get the error:
    mse = ((X_test - reconstructed) ** 2).sum(axis=1).sum() / X_test.shape[0]
    print("\nTest MSE: %.4f" % mse)
    # TODO: add assertions
