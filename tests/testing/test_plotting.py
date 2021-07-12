import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from synthesized.testing.plotting import plot_cross_tables
from synthesized.testing.plotting.series import (plot_categorical_time_series, plot_continuous_time_series, plot_series,
                                                 plot_time_series)


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


@pytest.fixture
def series_df():
    n = 500
    return pd.DataFrame({
        't': range(n),
        'x': np.random.randn(n),
        'y': np.random.randint(10, size=n),
        'z': np.random.choice(['a', 'b', 'c'], size=n),
    })


def test_plot_cross_tables(simple_df):
    df_orig, df_synth = train_test_split(simple_df, test_size=0.5)

    plot_cross_tables(df_orig, df_synth, 'string', 'int_bool')


def test_series_plotting(series_df):

    plot_series(series_df['x'], plt.subplots(1,1)[1])

    plot_time_series(series_df['x'], series_df['t'], plt.subplots(1, 1)[1])
    plot_time_series(series_df['y'], series_df['t'], plt.subplots(1, 1)[1])
    plot_time_series(series_df['z'], series_df['t'], plt.subplots(1, 1)[1])

    plot_continuous_time_series(series_df[:300], series_df[300:], 'x')
