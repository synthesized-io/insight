from typing import Union, List

import pandas as pd

from .base import Transformer


class BinningTransformer(Transformer):
    """
    Bin continous values into discrete bins.

    Attributes:
        name: the data frame column to transform.

        bins: the number of equally spaced bins or a predefined list of bin edges.

        **kwargs: keyword arguments to pd.cut

    See also:
        pd.cut
    """
    def __init__(self, name: str, bins: Union[List, int], **kwargs):
        super().__init__(name)
        self.bins = bins
        self.kwargs = kwargs

    def __repr__(self):
        return f'{self.__class__.__name__}(name="{self.name}", dtypes={self.dtypes}, bins={self.bins}, kwargs={repr(self.kwargs)}))'

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.name] = pd.cut(df[self.name], self.bins, **self.kwargs)
        return df


class QuantileBinningTransformer(Transformer):
    """
    Bin continous values into equal-size discrete bins based on quantiles of given data.

    Attributes:
        name: the data frame column to transform.

        quantiles: number of quantiles, or array of quantiles.

        **kwargs: keyword arguments to pd.qcut

    See also:
        pd.cut
    """
    def __init__(self, name: str, quantiles: Union[int, List[float]], **kwargs):
        super().__init__(name)
        self.quantiles = quantiles
        self.kwargs = kwargs

    def __repr__(self):
        return f'{self.__class__.__name__}(name="{self.name}", dtypes={self.dtypes}, quantiles={self.quantiles}, kwargs={repr(self.kwargs)}))'

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.name] = pd.qcut(df[self.name], self.quantiles, **self.kwargs)
        return df
