from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ..metadata import DataFrameMeta
from ..insight import metrics


MAX_PVAL = 0.05


def calculate_evaluation_metrics(df_orig: pd.DataFrame, df_synth: pd.DataFrame,
                                 df_meta: DataFrameMeta, column_names: Optional[List[str]] = None,
                                 metrics_to_compute: List[str] = None) -> Dict[str, Union[pd.Series, pd.DataFrame]]:
    """Calculate 'stop_metric' dictionary given two datasets. Each item in the dictionary will include a key
    (from self.stop_metric_name, allowed options are 'ks_dist', 'corr' and 'emd'), and a value (list of
    stop_metrics per column).

    Args
        df_orig: Original DataFrame.
        df_synth: Synthesized DataFrame.
        column_names: List of columns used to compute the 'break_metric'.

    Returns
        bool: True if criteria are met to stop learning.
    """
    if column_names is None:
        column_names_df: List[str] = df_orig.columns
    else:
        column_names_df = list(filter(lambda c: c in df_orig.columns, column_names))

    df_orig = df_orig.loc[:, column_names_df].copy()
    df_synth = df_synth.loc[:, column_names_df].copy()
    results = dict()

    # Calculate 1st order metrics for categorical/continuous
    if 'ks_distance' in metrics_to_compute:
        results['ks_distances'] = metrics.kolmogorov_smirnov_distance_vector(df_orig, df_synth, dp=df_meta)
    if 'emd_categ' in metrics_to_compute:
        results['emd_distances'] = metrics.earth_movers_distance_vector(df_orig, df_synth, dp=df_meta)

    # Calculate 2nd order metrics for categorical/continuous
    if 'corr_dist' in metrics_to_compute:
        results['corr_distances'] = np.abs(metrics.diff_kendell_tau_correlation_matrix(
            df_orig, df_synth, dp=df_meta
        ))
    if 'cramers_v' in metrics_to_compute:
        results['cramers_v_distances'] = np.abs(metrics.diff_cramers_v_matrix(df_orig, df_synth, dp=df_meta))
    if 'logistic_corr_dist' in metrics_to_compute:
        results['logistic_corr_distances'] = np.abs(metrics.diff_categorical_logistic_correlation_matrix(
            df_orig, df_synth, dp=df_meta, continuous_input_only=True
        ))

    return results
