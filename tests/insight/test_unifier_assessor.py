from typing import Sequence, Tuple

import pandas as pd
import pytest

from synthesized.insight.metrics import (CategoricalLogisticR2, CramersV, EarthMoversDistance, KendallTauCorrelation,
                                         KolmogorovSmirnovDistance)
from synthesized.insight.unifier import UnifierAssessor


@pytest.fixture(scope='module')
def tuple_of_df_sequence_and_df() -> Tuple[pd.DataFrame, Sequence[pd.DataFrame]]:
    """Generates multiple dataframes by random sampling some of the columns
       of credit.csv"""
    df = pd.read_csv("data/credit.csv", index_col=0).dropna().head(10000)
    df1 = df[['RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfOpenCreditLinesAndLoans']].sample(2000)
    df2 = df[['NumberOfDependents', 'age', 'MonthlyIncome']].sample(5000)
    df3 = df[['NumberOfDependents', 'RevolvingUtilizationOfUnsecuredLines', 'age']].sample(7500)
    df4 = df[['SeriousDlqin2yrs', 'RevolvingUtilizationOfUnsecuredLines',
              'age', 'DebtRatio', 'NumberOfDependents']].sample(8000)

    dfs = [df1, df2, df3, df4]
    unified_df = df[['SeriousDlqin2yrs', 'RevolvingUtilizationOfUnsecuredLines', 'age', 'MonthlyIncome',
                     'NumberOfDependents']].sample(7000)
    return dfs, unified_df


def test_first_order_metric_distances(tuple_of_df_sequence_and_df):
    dfs, unified_df = tuple_of_df_sequence_and_df
    unified_assessor = UnifierAssessor(dfs, unified_df)

    ks_dist_results = unified_assessor.get_first_order_metric_distances(KolmogorovSmirnovDistance())
    assert ks_dist_results is not None
    assert len(ks_dist_results) == len(dfs)

    em_dist_results = unified_assessor.get_first_order_metric_distances(EarthMoversDistance())
    assert em_dist_results is not None
    assert len(em_dist_results) == len(dfs)


def test_second_order_metric_matrices(tuple_of_df_sequence_and_df):
    dfs, unified_df = tuple_of_df_sequence_and_df
    unified_assessor = UnifierAssessor(dfs, unified_df)

    kdt_matrix_unified_df, kdt_matrix_orig_dfs_list = unified_assessor\
        .get_second_order_metric_matrices(KendallTauCorrelation())
    assert kdt_matrix_unified_df is not None
    assert len(kdt_matrix_orig_dfs_list) == len(dfs)

    cmv_matrix_unified_df, cmv_matrix_orig_dfs_list = unified_assessor\
        .get_second_order_metric_matrices(CramersV())
    assert cmv_matrix_unified_df is not None
    assert len(cmv_matrix_orig_dfs_list) == len(dfs)

    clr_matrix_unified_df, clr_matrix_orig_dfs_list = unified_assessor\
        .get_second_order_metric_matrices(CategoricalLogisticR2())
    assert clr_matrix_unified_df is not None
    assert len(clr_matrix_orig_dfs_list) == len(dfs)
