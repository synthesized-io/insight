from typing import Union, List

import pandas as pd

from .base import Transformer

# ToDo: No from_meta method yet. Though i feel like this class is very similar to the HistogramModel class.
#       (Maybe there is some subtle difference)
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

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        x[self.name] = pd.cut(x[self.name], self.bins, **self.kwargs)
        return x
