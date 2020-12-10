from typing import Union, List, Optional, cast

import numpy as np
import pandas as pd

from .base import Transformer


class BinningTransformer(Transformer):
    """
    Bin continous values into discrete bins.

    Attributes:
        name: the data frame column to transform.

        bins: the number of bins or a predefined list of bin edges (if not quantile based) or
            array of quantiles (if quantile based).
        remove_outliers: if given, proportion of outliers to remove before computing bins.
        quantile_based: whether the computed bins are quantile based
        **kwargs: keyword arguments to pd.cut (if not quantile based) or pd.qcut (if quantile based)

    See also:
        pd.cut, pd.qcut
    """
    def __init__(self, name: str, bins: Union[List, int], remove_outliers: Optional[float] = 0.1,
                 quantile_based: bool = False, **kwargs):
        super().__init__(name)
        self.bins = bins
        self.remove_outliers = remove_outliers
        self.quantile_based = quantile_based
        self.kwargs = kwargs

        self._bins: Optional[List] = None

    def __repr__(self):
        return (f'{self.__class__.__name__}(name={self.name}, dtypes={self.dtypes}, bins={self.bins}, '
               f'remove_outliers={self.remove_outliers}, quantile_based={self.quantile_based}, '
               f'{", ".join([f"{k}={v}" for k, v in self.kwargs.items()])})')

    def fit(self, df: pd.DataFrame) -> 'BinningTransformer':

        column = df[self.name].copy()
        column_clean = column.dropna()
        if self.remove_outliers:
            percentiles = [self.remove_outliers * 100. / 2, 100 - self.remove_outliers * 100. / 2]
            start, end = np.percentile(column_clean, percentiles)

            if start == end:
                start, end = min(column_clean), max(column_clean)

            column_clean = column_clean[(start <= column_clean) & (column_clean <= end)]

        if not self.quantile_based:
            _, bins = pd.cut(column_clean, self.bins, retbins=True, **self.kwargs)
        else:
            _, bins = pd.qcut(column_clean, self.bins, retbins=True, **self.kwargs)
        self._bins = list(bins)  # Otherwise it is np.ndarray
        self._bins[0], self._bins[-1] = column.min(), column.max()

        return super().fit(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._assert_fitted()

        df[self.name] = pd.cut(df[self.name], bins=self._bins, **self.kwargs)
        super().transform(df=df)
        return df
