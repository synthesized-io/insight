from typing import Dict, Sequence, Tuple, Union

import pandas as pd

from synthesized.insight.metrics import ColumnComparisonVector, TwoColumnMetric, TwoColumnMetricMatrix


class UnifierAssessor:
    """Class that lets compare the quality of unified data to the original data sources
    Attributes:
        sub_dfs: A Sequence of original dataframes
        unified_df: A unified dataframe created out of the original dfs
    """

    def __init__(self, sub_dfs: Union[Dict[str, pd.DataFrame], Sequence[pd.DataFrame]], unified_df: pd.DataFrame):
        """Create an instance of UtilityTesting."""
        self.sub_dfs: Dict[str, pd.DataFrame] = sub_dfs if isinstance(sub_dfs, dict) else {f"df{i}": df for i, df in enumerate(sub_dfs)}
        self.unified_df = unified_df

    def get_first_order_metric_distances(self, metric: TwoColumnMetric) -> Dict[str, pd.Series]:
        """Returns the comparison metrics of each column in the unified dataframe as compared
            to the distribution of that column in original dataframes

        Args:
            metric: The metric to be applied to compute distance

        Returns:
            results: A sequence of the series of the distance/metric
                     between the column in the unified dataframe and that column
                     in the original dataframe.
        """
        if metric.name is None:
            raise ValueError("Metric has no name.")
        metric_vector = ColumnComparisonVector(metric)
        results: Dict[str, pd.Series] = {}
        for name, df in self.sub_dfs.items():
            df = df[df.columns.intersection(self.unified_df.columns)]
            result: pd.Series = metric_vector(df_old=df, df_new=self.unified_df)
            results[name] = result.dropna()

        return results

    def get_filtered_metric_matrix(self, df, metric):
        """Removes those columns which are not used for computing correlation
        using the given metrics"""
        metric_matrix = TwoColumnMetricMatrix(metric)
        matrix = metric_matrix(df)

        for c in matrix.columns:
            if matrix.loc[:, c].isna().all() and matrix.loc[c, :].isna().all():
                matrix.drop(c, axis=1, inplace=True)
                matrix.drop(c, axis=0, inplace=True)
        return matrix

    def get_second_order_metric_matrices(self, metric: TwoColumnMetric) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Gets the correlation matrices for the unified data and the original dataframes

        Args:
            metric: The metric to be applied to compute correlation

        Returns:
            results: A tuple of correlation matrix of the unified dataframe columns and the sequence of
                     the correlation matrices of the original dataframes.
        """
        if metric.name is None:
            raise ValueError("Metric has no name.")

        matrix_sub_dfs_list = {}
        matrix_unified_df = self.get_filtered_metric_matrix(self.unified_df, metric)
        for name, df in self.sub_dfs.items():
            matrix_sub_dfs_list[name] = self.get_filtered_metric_matrix(df, metric)

        return matrix_unified_df, matrix_sub_dfs_list
