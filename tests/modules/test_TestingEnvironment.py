# -*- coding: utf-8 -*-
#
# Produced by: Synthesized Ltd
#
# Tests for TestingEnvironment


from __future__ import division, absolute_import, division

import os

import pandas as pd
import numpy as np

from synthesized.testing.testing_environment import TestingEnvironment


TESTDATA_ORIGINAL = os.path.join(os.path.dirname(__file__), '../../data/demo_original_dataset.csv')
TESTDATA_SYNTH = os.path.join(os.path.dirname(__file__), '../../data/demo_synthetic_dataset_perfect.csv')


def test_TestingEnvironment():
    testing_environment = TestingEnvironment()
    original_dataset, synthetic_dataset  = pd.read_csv(TESTDATA_ORIGINAL, sep='\t'), pd.read_csv(TESTDATA_SYNTH, sep='\t')
    testing_environment.compare_pred_performance(original_dataset, synthetic_dataset)