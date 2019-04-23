import pandas as pd


def compute_correlation_similarity(df_orig: pd.DataFrame, df_synth: pd.DataFrame) -> pd.DataFrame:
    corr_orig = df_orig.corr()
    corr_synth = df_synth.corr()
    diff_ratio = (corr_synth - corr_orig).abs() / corr_orig.abs()
    diff_ratio = diff_ratio.applymap(lambda x: min(x, 1.0))
    return 1 - diff_ratio
