import numpy as np
import pandas as pd
import pytest

from insight.check import ColumnCheck
from insight.metrics import CorrMatrix, CramersV, DiffCorrMatrix, EarthMoversDistance, TwoColumnMap


@pytest.fixture(scope="module")
def data():
    df = (
        pd.read_csv(
            "https://raw.githubusercontent.com/synthesized-io/datasets/master/tabular/templates/credit.csv"
        )
        .dropna()
        .reset_index(drop=True)
    )
    categorical_cols = []
    continuous_cols = []

    check = ColumnCheck()
    for col in df.columns:
        if check.continuous(df[col]):
            continuous_cols.append(col)
        else:
            categorical_cols.append(col)

    return df, categorical_cols, continuous_cols


def test_two_column_map(data):
    df, categorical_cols, continuous_cols = data[0], data[1], data[2]
    df1 = df.sample(1000).reset_index(drop=True)
    df2 = df.sample(1000).reset_index(drop=True)

    emd = EarthMoversDistance()

    col_map = TwoColumnMap(emd)
    emd_map_df = col_map(df1, df2)
    assert col_map.name == f"{str(emd)}_map"

    assert set(emd_map_df.columns.to_list()) == set(["metric_val"])
    assert all(not np.isnan(emd_map_df["metric_val"][cat]) for cat in categorical_cols)
    assert all(np.isnan(emd_map_df["metric_val"][cont]) for cont in continuous_cols)


def test_metric_matrix(data):
    df, categorical_cols, continuous_cols = data[0], data[1], data[2]
    df1 = df.sample(1000).reset_index(drop=True)
    df2 = df.sample(1000).reset_index(drop=True)

    cmv = CramersV()
    cmt = CorrMatrix(cmv)
    assert cmt.name == f"{str(cmv)}_matrix"
    cmv_val_df = cmt(df)
    assert all(
        np.isnan(cmv_val_df[cont1][cont2]) and np.isnan(cmv_val_df[cont2][cont1])
        for cont1 in continuous_cols
        for cont2 in continuous_cols
    )
    assert all(
        np.isnan(cmv_val_df[cat][cont]) and np.isnan(cmv_val_df[cont][cat])
        for cat in categorical_cols
        for cont in continuous_cols
    )
    assert all(
        not np.isnan(cmv_val_df[cat1][cat2]) and not np.isnan(cmv_val_df[cat2][cat1])
        for cat1 in categorical_cols
        for cat2 in categorical_cols
        if cat1 != cat2
    )

    cmv_diff_mat = DiffCorrMatrix(cmv)
    diff = cmv_diff_mat(df1, df2)
    assert cmv_diff_mat.name == f"diff_{str(cmv)}"
    assert all(
        np.isnan(diff[cont1][cont2]) and np.isnan(diff[cont2][cont1])
        for cont1 in continuous_cols
        for cont2 in continuous_cols
    )
    assert all(
        np.isnan(diff[cat][cont]) and np.isnan(diff[cont][cat])
        for cat in categorical_cols
        for cont in continuous_cols
    )
    assert all(
        not np.isnan(diff[cat1][cat2]) and not np.isnan(diff[cat2][cat1])
        for cat1 in categorical_cols
        for cat2 in categorical_cols
        if cat1 != cat2
    )
    assert not np.isnan(cmv_diff_mat.summarize_result(diff))
