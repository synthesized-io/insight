import numpy as np
import pandas as pd
import pytest

from synthesized_insight.check import ColumnCheck
from synthesized_insight.metrics import (
    ColumnComparisonVector,
    CramersV,
    DiffMetricMatrix,
    EarthMoversDistance,
    KolmogorovSmirnovDistanceTest,
    KruskalWallisTest,
    TwoColumnMetricMatrix,
)


@pytest.fixture(scope='module')
def data():
    df = pd.read_csv('tests/datasets/mini_credit.csv').dropna().reset_index(drop=True)
    categorical_cols = []
    continuous_cols = []

    check = ColumnCheck()
    for col in df.columns:
        if check.continuous(df[col]):
            continuous_cols.append(col)
        else:
            categorical_cols.append(col)

    return df, categorical_cols, continuous_cols


def test_column_comparison_vector(data):
    df, categorical_cols, continuous_cols = data[0], data[1], data[2]
    df1 = df.sample(1000).reset_index(drop=True)
    df2 = df.sample(1000).reset_index(drop=True)

    emd = EarthMoversDistance()
    ksdt = KolmogorovSmirnovDistanceTest()

    cv = ColumnComparisonVector(emd)
    emd_cv = cv(df1, df2)
    assert cv.name == f'{str(emd)}_vector'
    assert all(not np.isnan(emd_cv[cat]) for cat in categorical_cols)
    assert all(np.isnan(emd_cv[cont]) for cont in continuous_cols)

    cv = ColumnComparisonVector(ksdt)
    kdst_cv = cv(df1, df2)
    assert cv.name == f'{str(ksdt)}_vector'
    assert all(np.isnan(kdst_cv[cat]) for cat in categorical_cols)
    assert all(not np.isnan(kdst_cv[cont]) for cont in continuous_cols)


def test_metric_matrix(data):
    df, categorical_cols, continuous_cols = data[0], data[1], data[2]
    df1 = df.sample(1000).reset_index(drop=True)
    df2 = df.sample(1000).reset_index(drop=True)

    kwt = KruskalWallisTest()
    tm = TwoColumnMetricMatrix(kwt)
    assert tm.name == f'{str(kwt)}_matrix'
    kwt_tm = tm(df)
    assert all(not np.isnan(kwt_tm[cont1][cont2]) and not np.isnan(kwt_tm[cont2][cont1]) for cont1 in continuous_cols for cont2 in continuous_cols if cont1 != cont2)
    assert all(np.isnan(kwt_tm[cat][cont]) and np.isnan(kwt_tm[cont][cat]) for cat in categorical_cols for cont in continuous_cols)
    assert all(np.isnan(kwt_tm[cat1][cat2]) and np.isnan(kwt_tm[cat2][cat1]) for cat1 in categorical_cols for cat2 in categorical_cols)

    diff_mat = DiffMetricMatrix(tm)
    diff = diff_mat(df1, df2)
    assert diff_mat.name == f'diff_{str(kwt)}_matrix'
    assert all(not np.isnan(diff[cont1][cont2]) and not np.isnan(diff[cont2][cont1]) for cont1 in continuous_cols for cont2 in continuous_cols if cont1 != cont2)
    assert all(np.isnan(diff[cat][cont]) and np.isnan(diff[cont][cat]) for cat in categorical_cols for cont in continuous_cols)
    assert all(np.isnan(diff[cat1][cat2]) and np.isnan(diff[cat2][cat1]) for cat1 in categorical_cols for cat2 in categorical_cols)

    cmv = CramersV()
    tm = TwoColumnMetricMatrix(cmv)
    assert tm.name == f'{str(cmv)}_matrix'
    cmv_tm = tm(df)
    assert all(np.isnan(cmv_tm[cont1][cont2]) and np.isnan(cmv_tm[cont2][cont1]) for cont1 in continuous_cols for cont2 in continuous_cols)
    assert all(np.isnan(cmv_tm[cat][cont]) and np.isnan(cmv_tm[cont][cat]) for cat in categorical_cols for cont in continuous_cols)
    assert all(not np.isnan(cmv_tm[cat1][cat2]) and not np.isnan(cmv_tm[cat2][cat1]) for cat1 in categorical_cols for cat2 in categorical_cols if cat1 != cat2)

    diff_mat = DiffMetricMatrix(tm)
    diff = diff_mat(df1, df2)
    assert diff_mat.name == f'diff_{str(cmv)}_matrix'
    assert all(np.isnan(diff[cont1][cont2]) and np.isnan(diff[cont2][cont1]) for cont1 in continuous_cols for cont2 in continuous_cols)
    assert all(np.isnan(diff[cat][cont]) and np.isnan(diff[cont][cat]) for cat in categorical_cols for cont in continuous_cols)
    assert all(not np.isnan(diff[cat1][cat2]) and not np.isnan(diff[cat2][cat1]) for cat1 in categorical_cols for cat2 in categorical_cols if cat1 != cat2)
