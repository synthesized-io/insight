import numpy as np
import pandas as pd
from synthesized.highdim import HighDimSynthesizer
from synthesized.common.values.identify_rules import PairwiseRuleFactory
from synthesized.common.values.rule import RuleValue
from synthesized.common.values.identify_value import identify_value
from synthesized.common.values.identify_rules import identify_rules
import os
import pytest
import time

BASEDIR = os.path.dirname(__file__)
# There will be some error from setting the threshold to halfway between the boundaries of sets.
MAX_ERRORS = 2


def setup():
    pass


def test_piecewise_detection():
    # Create flags
    df = pd.DataFrame(np.random.randn(10**5, 20))
    df.loc[:, 15] = df.loc[:, 0] > 0.0
    df.loc[:, 16] = df.loc[:, 1] > 0.5
    df.loc[:, 17] = df.loc[:, 2] > 1
    df.loc[:, 18] = 1 * (df.loc[:, 3] > 1) - 1 * (df.loc[:,3] < -1)
    df.loc[:, 19] = 5 * (df.loc[:, 4] > 1) + 2 * (df.loc[:,4] < 0)
    df.columns = [str(x) for x in df.columns]
    synth = HighDimSynthesizer(df=df, find_rules=['find_piecewise'])
    assert len(synth.values) == 15
    for i in range(5):
        assert isinstance(synth.values[i], RuleValue)


@pytest.mark.integration
@pytest.mark.skip
def test_piecewise_generation():
    # Create flags
    df = pd.DataFrame(np.random.randn(10**5, 20))
    df.loc[:, 15] = df.loc[:, 0] > 0.0
    df.loc[:, 16] = df.loc[:, 1] > 0.5
    df.loc[df.loc[:, 2] > 1, 17] = 'apple'
    df.loc[df.loc[:, 2] <= 1, 17] = 'orange'
    df.loc[:, 18] = 1 * (df.loc[:, 3] > 1) - 1 * (df.loc[:,3] < -1)
    df.loc[:, 19] = 5 * (df.loc[:, 4] > 1) + 2 * (df.loc[:,4] < 0)
    df.columns = [str(x) for x in df.columns]
    with HighDimSynthesizer(df=df, find_rules=['find_piecewise']) as synthesizer:
        synthesizer.learn(df_train=df, num_iterations=200)
        synthesized = synthesizer.synthesize(num_rows=5000)

    # The threshold may not be exact. For a standard normal, this is quite compact, let us allow for 2 errors
    assert abs((synthesized['0'] > 0).sum() - synthesized['15'].sum()) <= MAX_ERRORS
    assert abs((synthesized['1'] > 0.5).sum() - synthesized['16'].sum()) <= MAX_ERRORS
    assert abs((synthesized['2'] > 1).sum() - (synthesized['17'] == 'apple').sum()) <= MAX_ERRORS
    assert abs((synthesized['2'] <= 1).sum() - (synthesized['17'] == 'orange').sum()) <= MAX_ERRORS
    assert abs((synthesized['3'] > 1).sum() - (synthesized['18'] == 1).sum()) <= MAX_ERRORS
    assert abs((synthesized['3'] < -1).sum() - (synthesized['18'] == -1).sum()) <= MAX_ERRORS
    assert abs((synthesized['4'] > 1).sum() - (synthesized['19'] == 5).sum()) <= MAX_ERRORS
    assert abs((synthesized['4'] < 0).sum() - (synthesized['19'] == 2).sum()) <= MAX_ERRORS


def test_pulse_detection():
    df = pd.DataFrame(np.random.randn(10**5, 20))
    df.loc[:, 15] = (df.loc[:, 0] > 0.0) & (df.loc[:, 0] < 1.0)
    df.loc[:, 16] = (df.loc[:, 1] < 0.0) | (df.loc[:, 1] > 1.0)
    df.loc[(df.loc[:, 2] > 0) & (df.loc[:, 2] < 1), 17] = 'apple'
    df.loc[(df.loc[:, 2] < 0) | (df.loc[:, 2] > 1), 17] = 'orange'
    df.loc[(df.loc[:, 3] > -0.5) & (df.loc[:, 3] < 0), 18] = 7
    df.loc[(df.loc[:, 3] < -0.5) | (df.loc[:, 3] > 0), 18] = 2
    df.columns = [str(x) for x in df.columns]
    synth = HighDimSynthesizer(df=df, find_rules=['find_pulse'])
    assert len(synth.values) == 16
    for i in range(4):
        assert isinstance(synth.values[i], RuleValue)


@pytest.mark.integration
@pytest.mark.skip
def test_pulse_generation():
    df = pd.DataFrame(np.random.randn(10**5, 20))
    df.loc[:, 15] = (df.loc[:, 0] > 0.0) & (df.loc[:, 0] < 1.0)
    df.loc[:, 16] = (df.loc[:, 1] < 0.0) | (df.loc[:, 1] > 1.0)
    df.loc[(df.loc[:, 2] > 0) & (df.loc[:, 2] < 1), 17] = 'apple'
    df.loc[(df.loc[:, 2] < 0) | (df.loc[:, 2] > 1), 17] = 'orange'
    df.loc[(df.loc[:, 3] > -0.5) & (df.loc[:, 3] < 0), 18] = 7
    df.loc[(df.loc[:, 3] < -0.5) | (df.loc[:, 3] > 0), 18] = 2
    df.columns = [str(x) for x in df.columns]

    with HighDimSynthesizer(df=df, find_rules=['find_pulse']) as synthesizer:
        synthesizer.learn(df_train=df, num_iterations=200)
        synthesized = synthesizer.synthesize(num_rows=5000)

    assert abs(((synthesized['0'] > 0) & (synthesized['0'] < 1)).sum() - synthesized['15'].sum()) <= MAX_ERRORS
    assert abs(((synthesized['1'] < 0) | (synthesized['1'] > 1)).sum() - synthesized['16'].sum()) <= MAX_ERRORS
    assert abs(((synthesized['2'] > 0) & (synthesized['2'] < 1)).sum() - (synthesized['17'] == 'apple').sum()) <= MAX_ERRORS
    assert abs(((synthesized['3'] > -.5) & (synthesized['3'] < 0)).sum() - (synthesized['18'] == 7).sum()) <= MAX_ERRORS


@pytest.mark.integration
@pytest.mark.parametrize('rule', PairwiseRuleFactory.continuous_categorical_tests +
                         PairwiseRuleFactory.continuous_categorical_tests +
                         PairwiseRuleFactory.categorical_categorical_tests)
@pytest.mark.skip
def test_times(rule):
    # The slowest possible case is when no rules are satisfied as then we search through all rules.
    df = pd.DataFrame(np.random.randn(10**6, 100))
    # Make lots of categorical variables
    df.loc[:, 50:80] = (df.loc[:, 50:80] > 0)
    df.loc[:, 80:98] = -1 * (df.loc[:, 80:98] < -1) + 1 * (df.loc[:, 80:98] > 1)
    df.loc[:, 99] = (df.loc[:,98] > 0)
    df.columns = [str(x) for x in df.columns]

    dummy = HighDimSynthesizer(df=pd.DataFrame(np.random.randn(5, 5)))

    # Make the list of values
    values = list()
    for name in df.columns:
        value = identify_value(module=dummy, df=df[name], name=name)
        assert len(value.columns()) == 1 and value.columns()[0] == name
        values.append(value)

    # Automatic extraction of specification parameters
    df = df.copy()
    for value in values:
        value.extract(df=df)

    # Identify deterministic rules
    start = time.time()
    values = identify_rules(values=values, df=df, tests='all')
    # Make sure it does not take longer than 2 minutes per test.
    assert time.time() - start < 120
