from typing import Sequence

import pandas as pd
import pytest
from synthesized.testing.unifier_modelling_assessor import UnifierModellingAssessor

@pytest.fixture(scope='module')
def original_df() -> pd.DataFrame:
    """Returns the dataframe of 50k randomly sampled rows of credit.csv"""
    return pd.read_csv("data/credit.csv", index_col=0).dropna().head(20000)

@pytest.fixture(scope='module')
def sequence_of_dfs(original_df) -> Sequence[pd.DataFrame]:
    """Generates multiple dataframes by random sampling some of the columns of given original df"""
    df1 = original_df[['RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfOpenCreditLinesAndLoans']].sample(2000)
    df2 = original_df[['NumberOfDependents', 'age', 'MonthlyIncome']].sample(5000)
    df3 = original_df[['NumberOfDependents', 'RevolvingUtilizationOfUnsecuredLines', 'age']].sample(7500)
    df4 = original_df[['SeriousDlqin2yrs', 'RevolvingUtilizationOfUnsecuredLines',
              'age', 'DebtRatio', 'NumberOfDependents']].sample(8000)

    dfs = [df1, df2, df3, df4]
    return dfs

@pytest.fixture(scope='module')
def unified_df(original_df) -> pd.DataFrame:
    """Simulated unified dataset: created by random sampling of rows of given original df"""
    unified_df = original_df[['SeriousDlqin2yrs', 'RevolvingUtilizationOfUnsecuredLines', 'age', 'MonthlyIncome',
                              'NumberOfDependents']].sample(7000)
    return unified_df
    
def test_unified_df_metric_wrt_original_df(unified_df, original_df):
    unifier_modelling = UnifierModellingAssessor(unified_df=unified_df, orig_df=original_df)
    unified_df_cols = unified_df.columns
    model = 'RandomForest'
    
    for col in unified_df_cols:
        target = col
        predictors = [pred for pred in unified_df_cols if pred!=target]
        _, scores = unifier_modelling.get_metric_score_for_unified_df(target=target,
                                                                      predictors=predictors,
                                                                      model=model)
        original_df_score, unified_df_score = scores[0], scores[1]
        assert original_df_score is not None and unified_df_score is not None
    

def test_unified_df_metric_wrt_original_dfs(unified_df, sequence_of_dfs):
    unifier_modelling = UnifierModellingAssessor(unified_df=unified_df, sub_dfs=sequence_of_dfs)
    unified_df_cols = unified_df.columns
    model = 'RandomForest'
    
    for col in unified_df_cols:
        target = col
        predictors = [pred for pred in unified_df_cols if pred!=target]
        _, scores_dict = unifier_modelling.get_metric_score_for_unified_df(target=target,
                                                                           predictors=predictors,
                                                                           model=model)
        assert scores_dict is not None
