import random
import string

import numpy as np
import pandas as pd
import pytest
from scipy.stats import ks_2samp

from synthesized import HighDimSynthesizer
from synthesized.common.rules import Association, Expression, ValueIsIn
from synthesized.common.values import DateValue
from synthesized.metadata import DataFrameMeta
from synthesized.metadata.factory import MetaExtractor
from synthesized.metadata.value import Float, Integer, String
from synthesized.model.models import Histogram, KernelDensityEstimate
from tests.utils import progress_bar_testing


@pytest.mark.slow
def test_continuous_variable_generation():
    r = np.random.normal(loc=5000, scale=1000, size=1000)
    df_original = pd.DataFrame({'r': r})
    df_meta = MetaExtractor.extract(df=df_original)
    synthesizer = HighDimSynthesizer(df_meta=df_meta)
    synthesizer.learn(num_iterations=1000, df_train=df_original)
    df_synthesized = synthesizer.synthesize(num_rows=len(df_original), progress_callback=progress_bar_testing)

    distribution_distance = ks_2samp(df_original['r'], df_synthesized['r'])[0]
    assert distribution_distance < 0.3


@pytest.mark.slow
def test_categorical_variable_generation():
    r = np.random.normal(loc=5, scale=1, size=1000)
    df_original = pd.DataFrame({'r': list(map(int, r))})
    df_meta = MetaExtractor.extract(df=df_original)
    synthesizer = HighDimSynthesizer(df_meta=df_meta)
    synthesizer.learn(num_iterations=1000, df_train=df_original)
    df_synthesized = synthesizer.synthesize(num_rows=len(df_original), progress_callback=progress_bar_testing)

    distribution_distance = ks_2samp(df_original['r'], df_synthesized['r'])[0]
    assert distribution_distance < 0.3


@pytest.mark.slow
def test_date_variable_generation():
    df_original = pd.DataFrame({
        'z': pd.date_range(start='01/01/1900', end='01/01/2020', periods=1000).strftime("%d/%m/%Y"),
    }).sample(frac=1.)
    df_meta = MetaExtractor.extract(df=df_original)
    synthesizer = HighDimSynthesizer(df_meta=df_meta)
    synthesizer.learn(num_iterations=1000, df_train=df_original)
    df_synthesized = synthesizer.synthesize(num_rows=len(df_original), progress_callback=progress_bar_testing)

    assert isinstance(synthesizer.df_value['z'], DateValue)


@pytest.mark.slow
def test_nan_producing():
    size = 1000
    nans_prop = 0.33

    df_original = pd.DataFrame({
        'x1': np.random.normal(loc=0, scale=1, size=size),
        'x2': np.random.normal(loc=0, scale=1, size=size),
        'y1': np.random.choice(['A', 'B'], size=size),
        'y2': np.random.choice(['A', 'B'], size=size),
    })
    df_original.loc[np.random.uniform(size=len(df_original)) < nans_prop, 'x2'] = np.nan
    df_original.loc[np.random.uniform(size=len(df_original)) < nans_prop, 'y2'] = np.nan

    df_meta = MetaExtractor.extract(df=df_original)
    synthesizer = HighDimSynthesizer(df_meta=df_meta)
    synthesizer.learn(num_iterations=100, df_train=df_original)
    df_synthesized = synthesizer.synthesize(num_rows=len(df_original), produce_nans=True,
                                            progress_callback=progress_bar_testing)
    assert df_synthesized['x1'].isna().sum() == 0
    assert df_synthesized['y1'].isna().sum() == 0
    assert df_synthesized['x2'].isna().sum() > 0
    assert df_synthesized['y2'].isna().sum() > 0

    df_synthesized = synthesizer.synthesize(num_rows=len(df_original), produce_nans=False,
                                            progress_callback=progress_bar_testing)
    assert df_synthesized['x1'].isna().sum() == 0
    assert df_synthesized['y1'].isna().sum() == 0
    assert df_synthesized['x2'].isna().sum() == 0
    assert df_synthesized['y2'].isna().sum() == 0


@pytest.mark.slow
def test_sampling():
    size = 1000

    df_original = pd.DataFrame({
        'x1': np.random.normal(loc=0, scale=1, size=size),
        'x2': np.random.normal(loc=0, scale=1, size=size),
        'y1': np.random.choice(['A', 'B'], size=size),
        'y2': np.random.choice(['A', 'B'], size=size),
        'sample': [''.join([random.choice(string.ascii_letters) for _ in range(10)]) for _ in range(size)]

    })

    df_meta = MetaExtractor.extract(df=df_original)
    synthesizer = HighDimSynthesizer(df_meta=df_meta)
    synthesizer.learn(num_iterations=10, df_train=df_original)
    df_synthesized = synthesizer.synthesize(num_rows=len(df_original), progress_callback=progress_bar_testing)

    assert all([c in synthesizer.df_model for c in ['x1', 'x2', 'y1', 'y2']])
    assert 'sample' in synthesizer.df_model_independent


@pytest.mark.slow
def test_sampling_nans():
    size = 1000
    nans_prop = 0.33

    df_original = pd.DataFrame({
        'x1': np.random.normal(loc=0, scale=1, size=size),
        'x2': np.random.normal(loc=0, scale=1, size=size),
        'y1': np.random.choice(['A', 'B'], size=size),
        'y2': np.random.choice(['A', 'B'], size=size),
        'sample': [''.join([random.choice(string.ascii_letters) for _ in range(10)]) for _ in range(size)]
    })
    df_original['sample'] = np.where(
        df_original.index.isin(np.random.choice(df_original.index, size=int(len(df_original) * nans_prop))), np.nan,
        df_original['sample'])

    df_meta = MetaExtractor.extract(df=df_original)
    synthesizer = HighDimSynthesizer(df_meta=df_meta)
    synthesizer.learn(num_iterations=10, df_train=df_original)

    assert all([c in synthesizer.df_model for c in ['x1', 'x2', 'y1', 'y2']])
    assert 'sample' in synthesizer.df_model_independent

    df_synthesized = synthesizer.synthesize(num_rows=len(df_original), produce_nans=False,
                                            progress_callback=progress_bar_testing)
    assert df_synthesized['sample'].isna().sum() == 0

    df_synthesized = synthesizer.synthesize(num_rows=len(df_original), produce_nans=True,
                                            progress_callback=progress_bar_testing)
    assert df_synthesized['sample'].isna().sum() > 0


@pytest.mark.slow
@pytest.mark.parametrize("use_generic_rule", [False, True])
@pytest.mark.parametrize("use_expression_rule", [False, True])
@pytest.mark.parametrize("use_association_rule", [False, True])
def test_synthesize_with_rules(use_generic_rule, use_expression_rule, use_association_rule):
    size = 1000
    nans_prop = 0.33

    df_original = pd.DataFrame({
        'x1': np.random.normal(loc=0, scale=1, size=size),
        'x2': np.random.normal(loc=0, scale=1, size=size),
        'y1': np.random.choice(['A', 'B'], size=size),
        'y2': np.random.choice(['A', 'B'], size=size),
        'sample': [''.join([random.choice(string.ascii_letters) for _ in range(10)]) for _ in range(size)]
    })

    df_original.loc[np.random.uniform(size=len(df_original)) < nans_prop, 'x1'] = np.nan

    df_meta = MetaExtractor.extract(df=df_original)
    synthesizer = HighDimSynthesizer(df_meta=df_meta)
    synthesizer.learn(num_iterations=10, df_train=df_original)

    generic_rules = [ValueIsIn("y1", ["A"])] if use_generic_rule else []

    binding_mask = np.array([[1, 0], [0, 1]]) if df_meta["y2"].categories[0] == "A" else np.array([[0, 1], [1, 0]])
    association_rules = [Association(binding_mask=binding_mask, associations=["y2"], nan_associations=["x1"])] if use_association_rule else []

    expression_rules = [Expression("x3", "2 * x2")] if use_expression_rule else []

    df_synthesized = synthesizer.synthesize_from_rules(num_rows=len(df_original), progress_callback=progress_bar_testing,
                                                       generic_rules=generic_rules, association_rules=association_rules,
                                                       expression_rules=expression_rules, produce_nans=True)

    if use_expression_rule:
        assert (df_synthesized["x3"] == 2 * df_synthesized["x2"]).all()
    if use_generic_rule:
        assert (df_synthesized["y1"] == "A").all()
    if use_association_rule:
        first_case = (df_synthesized["y2"] == "A") * (~df_synthesized["x1"].isna())
        second_case = (df_synthesized["y2"] == "B") * (df_synthesized["x1"].isna())
        assert (first_case + second_case).all()

    assert len(df_synthesized) == len(df_original)


@pytest.mark.slow
def test_synthesize_bad_generic_rules_raises_error():
    size = 1000

    df_original = pd.DataFrame({
        'y1': np.random.choice(['A', 'B'], size=size),
    })

    df_meta = MetaExtractor.extract(df=df_original)
    synthesizer = HighDimSynthesizer(df_meta=df_meta)
    synthesizer.learn(num_iterations=10, df_train=df_original)

    generic_rules = [ValueIsIn("y1", ["C"])]
    with pytest.raises(RuntimeError):
        synthesizer.synthesize_from_rules(num_rows=len(df_original), generic_rules=generic_rules)


@pytest.mark.slow
def test_inf_not_producing():
    r = np.random.normal(loc=0, scale=1, size=1000)
    df_original = pd.DataFrame({'r': r}, dtype=np.float32)
    indices = np.random.choice(np.arange(r.size), replace=False, size=int(r.size * 0.1))
    df_original.iloc[indices] = np.inf
    indices = np.random.choice(np.arange(r.size), replace=False, size=int(r.size * 0.1))
    df_original.iloc[indices] = -np.inf
    df_meta = MetaExtractor.extract(df=df_original)
    synthesizer = HighDimSynthesizer(df_meta=df_meta)
    synthesizer.learn(num_iterations=1000, df_train=df_original)
    df_synthesized = synthesizer.synthesize(num_rows=len(df_original), progress_callback=progress_bar_testing)
    assert df_synthesized['r'].isin([np.Inf, -np.Inf]).sum() == 0


@pytest.mark.slow
def test_type_overrides():
    n = 1000
    df_original = pd.DataFrame({
        'r1': np.random.randint(1, 5, size=n),
        'r2': np.random.randint(1, 5, size=n),
    })

    df_meta = MetaExtractor.extract(df=df_original)

    type_overrides = [
        KernelDensityEstimate(df_meta["r1"]),
        Histogram(df_meta["r2"])
    ]

    synthesizer = HighDimSynthesizer(df_meta=df_meta, type_overrides=type_overrides)
    synthesizer.learn(df_original, num_iterations=10)

    assert isinstance(synthesizer.df_model['r1'], KernelDensityEstimate)
    assert isinstance(synthesizer.df_model['r2'], Histogram)


@pytest.mark.slow
def test_encode():
    n = 1000
    df_original = pd.DataFrame({'x': np.random.normal(size=n), 'y': np.random.choice(['a', 'b', 'c'], size=n)})
    df_meta = MetaExtractor.extract(df=df_original)
    synthesizer = HighDimSynthesizer(df_meta=df_meta)
    synthesizer.learn(num_iterations=10, df_train=df_original)
    _, df_synthesized = synthesizer.encode(df_original)
    assert df_synthesized.shape == df_original.shape


@pytest.mark.slow
def test_encode_unlearned_meta():
    n = 1000
    df_original = pd.DataFrame({'x': np.random.normal(size=n), 'y': np.random.choice(['a', 'b', 'c'], size=n), 'z': np.full(n, 1.0)})

    x = Float('x')
    x.extract(df_original)

    y = String('y')
    y.extract(df_original)

    z = Integer('z')
    z.extract(df_original)

    df_meta = DataFrameMeta(name='df_meta')
    for meta in [x, y, z]:
        df_meta[meta.name] = meta

    synthesizer = HighDimSynthesizer(df_meta=df_meta)
    synthesizer.learn(num_iterations=10, df_train=df_original)
    _, df_synthesized = synthesizer.encode(df_original)
    assert df_synthesized.shape == df_original.shape


@pytest.mark.slow
def test_encode_deterministic():
    n = 1000
    df_original = pd.DataFrame({'x': np.random.normal(size=n), 'y': np.random.choice(['a', 'b', 'c'], size=n)})
    df_meta = MetaExtractor.extract(df=df_original)
    synthesizer = HighDimSynthesizer(df_meta=df_meta)
    synthesizer.learn(num_iterations=10, df_train=df_original)
    df_synthesized = synthesizer.encode_deterministic(df_original)
    assert df_synthesized.shape == df_original.shape
