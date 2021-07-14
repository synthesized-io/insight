import numpy as np
import pandas as pd

from synthesized.metadata import ValueMeta
from synthesized.transformer.base import Transformer
from synthesized.transformer.exceptions import NonInvertibleTransformError


class RoundingTransformer(Transformer):
    """
    Tranforms by binning the numerical column to 20 (or N) bins.
    Examples:
        102,845 -> "[100,000, 125,000)"

    Attributes:
        name (str): the data frame column to transform.
        n_bins (int): number of bins, default value is 20
    """

    def __init__(self, name: str, n_bins: int = 20):
        super().__init__(name)
        self.n_bins = n_bins
        self.bins: np.array = None

    def __repr__(self):
        return f'{self.__class__.__name__}(name="{self.name}")'

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fits the given dataframe to the transformer

        Args:
            df: Dataset to fit

        Returns:
            self
        """
        col_name = self.name
        column = pd.to_numeric(df[col_name], errors='coerce')
        if pd.to_numeric(df[col_name], errors='coerce').isna().all():
            raise ValueError(f"Can apply rounding transformer to column '{self.name}' "
                             f"as it doesn't contain numerical values.")

        _, self.bins = pd.qcut(column, q=self.n_bins, retbins=True, duplicates='drop')
        return super().fit(df)

    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Transforms the given dataframe using fitted transformer

        Args:
            df: Dataset to transform

        Returns:
            Transformed dataset
        """
        col_name = self.name
        df.loc[:, col_name] = pd.to_numeric(df.loc[:, col_name], errors='coerce')
        df.loc[:, col_name] = pd.cut(df.loc[:, col_name], bins=self.bins, include_lowest=True,
                                     duplicates='drop').astype(str)
        return df

    def inverse_transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        raise NonInvertibleTransformError

    @classmethod
    def from_meta(cls, meta: ValueMeta) -> 'RoundingTransformer':
        return cls(meta.name)
