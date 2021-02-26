import logging
from typing import List, Optional, cast

import pandas as pd

from ...metadata_new import ContinuousModel, DataFrameMeta, DateTime, MetaExtractor, Scale, ValueMeta
from ...metadata_new.model import ModelFactory
from ...transformer import BinningTransformer, DTypeTransformer, SequentialTransformer, Transformer

logger = logging.getLogger(__name__)


class FairnessTransformer(SequentialTransformer):
    """
    Fairness transformer

    Attributes:
        sensitive_attrs: List of columns containing sensitive attributes.
        target: Target variable to compute biases against.
        n_bins: Number of bins for sensitive attributes to be binned.
        target_n_bins: Number of bins for target to be binned, if None will use it as it is.
        positive_class: The sign of the biases depends on this class (positive biases have higher rate of this
            class). If not given, minority class will be used. Only used for binomial target variables.
        drop_dates: Whether to ignore sensitive attributes containing dates.
    """

    def __init__(self, sensitive_attrs: List[str], target: str, df_meta: Optional[DataFrameMeta] = None,
                 df_models: DataFrameMeta = None, n_bins: int = 5, target_n_bins: Optional[int] = 5,
                 positive_class: Optional[str] = None):

        self.df_meta = df_meta
        self.df_models = df_models
        self.sensitive_attrs = sensitive_attrs
        self.target = target
        self.n_bins = n_bins
        self.target_n_bins = target_n_bins
        self._used_columns = self.sensitive_attrs + [self.target]

        super().__init__(name="fairness_transformer")

    def fit(self, df: pd.DataFrame) -> 'FairnessTransformer':

        df = self._get_dataframe_subset(df)

        if len(df) == 0:
            logger.warning("Empty DataFrame.")
            return self

        if self.df_meta is None:
            self.df_meta = MetaExtractor.extract(df)

        if self.df_models is None:
            models = ModelFactory()(self.df_meta)
            assert isinstance(models, DataFrameMeta)
            self.df_models = models

        # Transformer for target column
        if isinstance(self.df_models[self.target], ContinuousModel) and self.target_n_bins:
            meta = cast(ValueMeta, self.df_meta[self.target])
            df = DTypeTransformer(self.target, meta.dtype).fit_transform(df)
            self.append(BinningTransformer(self.target, bins=self.target_n_bins, duplicates='drop',
                        remove_outliers=0.1, include_lowest=True))
        self.append(DTypeTransformer(self.target, out_dtype='str'))

        # Transformers for sensitive columns
        for col in self.sensitive_attrs:

            meta = cast(ValueMeta, self.df_meta[col])
            df = DTypeTransformer(col, meta.dtype).fit_transform(df)

            if isinstance(self.df_models[col], ContinuousModel):
                transformer = self._get_sensitive_attr_transformer(meta, col)
                if transformer:
                    self.append(transformer)

            # We want to always convert to string otherwise grouping operations can fail
            to_str_transformer = DTypeTransformer(col, out_dtype='str')
            self.append(to_str_transformer)

        super().fit(df)
        return self

    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:

        df = self._get_dataframe_subset(df)

        if len(df) == 0:
            logger.warning("Empty DataFrame.")
            return df

        return super().transform(df)

    def _get_sensitive_attr_transformer(self, meta: ValueMeta, column_name: str) -> Optional[Transformer]:

        if isinstance(meta, Scale):
            return BinningTransformer(column_name, bins=self.n_bins, duplicates='drop',
                                      include_lowest=True, remove_outliers=0.1)
        elif isinstance(meta, DateTime):
            return BinningTransformer(column_name, bins=self.n_bins, remove_outliers=None,
                                      duplicates='drop', include_lowest=True)
        else:
            return None

    def _get_dataframe_subset(self, df: pd.DataFrame) -> pd.DataFrame:
        if not all(col in df.columns for col in self._used_columns):
            raise KeyError("Target variable or sensitive attributes not present in DataFrame.")

        df = df[self._used_columns].copy()
        return df[~df[self.target].isna()]
