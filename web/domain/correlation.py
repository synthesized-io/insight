import pandas as pd


def compute_correlation_similarity(df_orig: pd.DataFrame, df_synth: pd.DataFrame, columns) -> pd.DataFrame:
    corr_orig = df_orig[columns].corr()
    corr_synth = df_synth[columns].corr()
    return 1 - (corr_synth - corr_orig).abs() / corr_orig.abs()
