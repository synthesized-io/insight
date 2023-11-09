import pandas as pd
import pytest

from insight.plot import categorical, continuous, cross_table, cross_tables, dataset, text_only


@pytest.fixture(scope="module")
def df():
    df = pd.read_csv(
        "https://raw.githubusercontent.com/synthesized-io/datasets/master/tabular/templates/adult.csv"
    )
    return df


@pytest.fixture(scope="module")
def df_half_a(df):
    df_half_a = df.iloc[: len(df) // 2, :]
    return df_half_a


@pytest.fixture(scope="module")
def df_half_b(df):
    df_half_b = df.iloc[len(df) // 2 :, :]
    return df_half_b


def test_plot_two_cross_tables_not_equal_not_none(df_half_a, df_half_b):
    categories_a = pd.concat((df_half_a["workclass"], df_half_b["workclass"])).unique()
    categories_b = pd.concat((df_half_a["marital-status"], df_half_b["marital-status"])).unique()
    categories_a.sort()
    categories_b.sort()

    fig_a = cross_table(
        pd.crosstab(
            pd.Categorical(df_half_a["workclass"], categories_a, ordered=True),
            pd.Categorical(df_half_b["marital-status"], categories_b, ordered=True),
            dropna=False,
        ),
        title="test",
    )

    assert fig_a is not None

    fig_b = cross_table(
        pd.crosstab(
            pd.Categorical(df_half_a["workclass"], categories_a, ordered=True),
            pd.Categorical(df_half_b["workclass"], categories_a, ordered=True),
            dropna=False,
        ),
        title="test",
    )
    assert fig_b is not None

    assert fig_a != fig_b


def test_plot_two_pairs_cross_tables_not_equal_not_none(df_half_a, df_half_b):
    fig_a = cross_tables(df_half_a, df_half_b, col_a="workclass", col_b="income")
    assert fig_a is not None

    fig_b = cross_tables(df_half_a, df_half_b, col_a="marital-status", col_b="income")
    assert fig_b is not None

    assert fig_a != fig_b


def test_plot_text_only_not_none(df_half_a, df_half_b):
    fig = text_only("This is the text")
    assert fig is not None


def test_plot_two_categorical_distribution_not_equal_not_none(df_half_a, df_half_b):
    fig_a = categorical([df_half_a["workclass"], df_half_b["workclass"]])
    assert fig_a is not None

    fig_b = categorical([df_half_a["workclass"]])
    assert fig_b is not None

    assert fig_a != fig_b


def test_plot_two_continuous_not_equal_not_none(df_half_a, df_half_b):
    fig_a = continuous([df_half_a["fnlwgt"], df_half_b["fnlwgt"]])
    assert fig_a is not None

    fig_b = continuous([df_half_a["fnlwgt"]])
    assert fig_b is not None

    assert fig_a != fig_b


def test_plot_dataset_two_datasets_not_equal_not_none(df_half_a, df_half_b):
    fig_a = dataset([df_half_a, df_half_b], max_categories=10000)
    assert fig_a is not None

    fig_b = dataset([df_half_a, df_half_b])
    assert fig_b is not None

    assert fig_a != fig_b
