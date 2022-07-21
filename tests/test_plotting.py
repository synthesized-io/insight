import pandas as pd
import pytest

from synthesized_insight.plotting import (
    categorical_distribution_plot,
    continuous_distribution_plot,
    plot_cross_table,
    plot_cross_tables,
    plot_dataset,
    plot_text_only,
)


@pytest.fixture(scope='module')
def df():
    df = pd.read_csv('https://raw.githubusercontent.com/synthesized-io/datasets/master/tabular/templates/adult.csv')
    return df


@pytest.fixture(scope='module')
def df_half_a(df):
    df_half_a = df.iloc[:len(df) // 2, :]
    return df_half_a


@pytest.fixture(scope="module")
def df_half_b(df):
    df_half_b = df.iloc[len(df) // 2:, :]
    return df_half_b


def test_plot_cross_table(df_half_a, df_half_b):
    categories_a = pd.concat((df_half_a['workclass'], df_half_b['workclass'])).unique()
    categories_b = pd.concat((df_half_a['marital-status'], df_half_b['marital-status'])).unique()
    categories_a.sort()
    categories_b.sort()

    fig_a = plot_cross_table(
        pd.crosstab(
            pd.Categorical(df_half_a['workclass'], categories_a, ordered=True),
            pd.Categorical(df_half_b['marital-status'], categories_b, ordered=True),
            dropna=False
        ),
        title='test'
    );

    assert fig_a is not None

    fig_b = plot_cross_table(
        pd.crosstab(
            pd.Categorical(df_half_a['workclass'], categories_a, ordered=True),
            pd.Categorical(df_half_b['workclass'], categories_a, ordered=True),
            dropna=False
        ),
        title='test'
    );
    assert fig_b is not None

    assert fig_a != fig_b


def test_plot_cross_tables(df_half_a, df_half_b):
    fig_a = plot_cross_tables(df_half_a, df_half_b, col_a='workclass', col_b='income')
    assert fig_a is not None

    fig_b = plot_cross_tables(df_half_a, df_half_b, col_a='marital-status', col_b='income')
    assert fig_b is not None

    assert fig_a != fig_b


def test_plot_text_only(df_half_a, df_half_b):
    fig = plot_text_only("This is the text")
    assert fig is not None


def test_plot_categorical_distribution(df_half_a, df_half_b):
    fig_a = categorical_distribution_plot([df_half_a['workclass'], df_half_b['workclass']])
    assert fig_a is not None

    fig_b = categorical_distribution_plot([df_half_a['workclass']])
    assert fig_b is not None

    assert fig_a != fig_b


def test_continuous_distribution_plot(df_half_a, df_half_b):
    fig_a = continuous_distribution_plot([df_half_a['fnlwgt'], df_half_b['fnlwgt']])
    assert fig_a is not None

    fig_b = continuous_distribution_plot([df_half_a['fnlwgt']])
    assert fig_b is not None

    assert fig_a != fig_b


def test_plot_dataset(df_half_a, df_half_b):
    fig_a = plot_dataset([df_half_a, df_half_b], max_categories=10000)
    assert fig_a is not None

    fig_b = plot_dataset([df_half_a, df_half_b])
    assert fig_b is not None

    assert fig_a != fig_b
