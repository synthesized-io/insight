import pandas as pd


def compute_correlation_similarity(df_orig: pd.DataFrame, df_synth: pd.DataFrame) -> pd.DataFrame:
    corr_orig = df_orig.corr()
    corr_synth = df_synth.corr()
    diff = ((corr_synth - corr_orig) / 2.).abs()
    return 1 - diff
