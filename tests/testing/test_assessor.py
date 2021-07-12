import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from synthesized.metadata.factory import MetaExtractor
from synthesized.testing import Assessor


@pytest.fixture
def simple_df():
    np.random.seed(6235901)
    n = 1000
    df = pd.DataFrame({
        'string': np.random.choice(['A','B','C','D','E'], size=n),
        'bool': np.random.choice([False, True], size=n).astype('?'),
        'date': pd.to_datetime(18_000 + np.random.normal(500, 50, size=n).astype(int), unit='D'),
        'int': np.random.choice([0, 1, 2, 3, 4, 5], size=n),
        'float': np.random.normal(0.0, 1.0, size=n),
        'float-big': np.random.normal(0.0, 10000.0, size=n),
        'int_bool': np.random.choice([0, 1], size=n),
        'date_sparse': pd.to_datetime(18_000 + 5 * np.random.normal(500, 50, size=n).astype(int), unit='D')
    })
    return df


def test_assessor_standard_metrics(simple_df):
    df_meta = MetaExtractor.extract(simple_df)
    assessor = Assessor(df_meta)

    df_orig, df_synth = train_test_split(simple_df.sample(200), test_size=0.5)

    assessor.show_standard_metrics(df_orig, df_synth, ax=None)


def test_assessor_marginals(simple_df):
    df_meta = MetaExtractor.extract(simple_df)
    assessor = Assessor(df_meta)

    df_orig, df_synth = train_test_split(simple_df.sample(200), test_size=0.5)

    assessor.show_distributions(df_orig, df_synth, remove_outliers=0.01)
    assessor.show_ks_distances(df_orig, df_synth)
    assessor.show_emd_distances(df_orig, df_synth)


def test_assessor_correlation_matrices(simple_df):
    df_meta = MetaExtractor.extract(simple_df)
    assessor = Assessor(df_meta)

    df_orig, df_synth = train_test_split(simple_df.sample(200), test_size=0.5)

    assessor.show_kendall_tau_matrices(df_orig, df_synth)
    assessor.show_spearman_rho_matrices(df_orig, df_synth)
    assessor.show_cramers_v_matrices(df_orig, df_synth)
    assessor.show_categorical_logistic_r2_matrices(df_orig, df_synth)


def test_assessor_correlation_distances(simple_df):
    df_meta = MetaExtractor.extract(simple_df)
    assessor = Assessor(df_meta)

    df_orig, df_synth = train_test_split(simple_df.sample(200), test_size=0.5)

    assessor.show_kendall_tau_distances(df_orig, df_synth)
    assessor.show_spearman_rho_distances(df_orig, df_synth)
    assessor.show_cramers_v_distances(df_orig, df_synth)
    assessor.show_categorical_logistic_r2_distances(df_orig, df_synth)


def test_assessor_modelling(simple_df):
    df_meta = MetaExtractor.extract(simple_df)
    assessor = Assessor(df_meta)

    df_orig, df_synth = train_test_split(simple_df.sample(200), test_size=0.5)

    assessor.plot_classification_metrics(df_orig, df_synth, target="int_bool", df_test=df_orig, clf=LogisticRegression())
    assessor.plot_classification_metrics_test(df_orig, df_synth.sample(100), df_synth.sample(100), target="int_bool", clf=LogisticRegression())
    assessor.utility(df_orig, df_synth, target="int_bool", model="Linear")
