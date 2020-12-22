import logging
from enum import Enum
from math import log
from typing import List, Optional, cast

import numpy as np
import pandas as pd

from ...transformer import Transformer, SequentialTransformer, DTypeTransformer, BinningTransformer
from ...config import MetaExtractorConfig
from ...metadata_new import DataFrameMeta, MetaExtractor, Scale, Date, ValueMeta

logger = logging.getLogger(__name__)


class VariableType(Enum):
    Binary = 0
    Multinomial = 1
    Continuous = 2


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

    def __init__(self, sensitive_attrs: List[str], target: str, df_meta: Optional[DataFrameMeta] = None, n_bins: int = 5,
                 target_n_bins: Optional[int] = 5, positive_class: Optional[str] = None):

        self.df_meta: Optional[DataFrameMeta] = None
        self.sensitive_attrs = sensitive_attrs
        self.target = target
        self.n_bins = n_bins
        self.target_n_bins = target_n_bins
        self._used_columns = self.sensitive_attrs + [self.target]

        super().__init__(name="fairness_transformer")

    def fit(self, df: pd.DataFrame) -> 'FairnessTransformer':

        df = self._get_dataframe_subset(df)

        if len(df) == 0:
            return self

        if self.df_meta is None:
            self.df_meta = MetaExtractor.extract(df)

        # To do: remove. This should be handled by Model, and the logic in MetaBuilder.
        categorical_threshold = int(max(
            float(MetaExtractorConfig.min_num_unique),
            MetaExtractorConfig.categorical_threshold_log_multiplier * log(len(df))
        ))

        # Transformer for target column
        if df[self.target].nunique() > categorical_threshold and self.target_n_bins:
            self.append(BinningTransformer(self.target, bins=self.target_n_bins, duplicates='drop', include_lowest=True))

        # Transformers for sensitive columns
        for col in self.sensitive_attrs:

            meta = cast(ValueMeta, self.df_meta[col])
            dtype_transformer = DTypeTransformer(col, meta.dtype)
            self.append(dtype_transformer)

            _categorical_threshold = self.n_bins if categorical_threshold is None else categorical_threshold
            num_unique = df[col].nunique()

            if num_unique > _categorical_threshold:
                transformer = self._get_sensitive_attr_transformer(meta, col)
                if transformer:
                    self.append(transformer)


            # If it's datetime, bin it
            # to do: move to FairnessScorer
            # if isinstance(meta, Date):
            #     if self.drop_dates:
            #         self.sensitive_attrs.remove(col)
            #         logging.info(f"Sensitive attribute '{col}' dropped as it is a date value and 'drop_dates=True'.")
            #         continue
            #     else:
            #         transformer = BinningTransformer(col, bins=self.n_bins, remove_outliers=None,
            #                                          duplicates='drop', include_lowest=True)

            #     if col == self.target:
            #         raise TypeError("Datetime target columns not supported")

            # If it's a sampling value, discard it
            # to do: move to FairnessScorer
            # elif num_unique > np.sqrt(len(df)):
            #     if col == self.target:
            #         raise TypeError("Target column has too many unique non-numerical values to compute fairness.")

            #     else:
            #         self.sensitive_attrs.remove(col)
            #         logging.info(f"Sensitive attribute '{col}' dropped as it is a sampled value.")
            #         continue

            # if transformer is not None:
            #     self.append(transformer)

            # We want to always convert to string otherwise grouping operations can fail
            to_str_transformer = DTypeTransformer(col, out_dtype='str')
            self.append(to_str_transformer)

        super().fit(df)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        df = self._get_dataframe_subset(df)

        if len(df) == 0:
            return df

        return super().transform(df)

    # to do: move to FairnessScorer
    # def get_positive_class(self, df: pd.DataFrame) -> Optional[str]:
    #     # Only set positive class for binary/multinomial, even if given.
    #     df_target = df[[self.target]].copy()
    #     for transformer in [t for t in self if t.name == self.target]:
    #         df_target = transformer.transform(df_target)

    #     target_column = df_target[self.target]

    #     if self.target_variable_type in (VariableType.Binary, VariableType.Multinomial):
    #         target_vc = target_column.value_counts(normalize=True)
    #         if len(target_vc) <= 2:
    #             if self.positive_class is None:
    #                 # If target class is not given, we'll use minority class as usually it is the target.
    #                 return str(target_vc.idxmin())
    #             elif self.positive_class not in target_vc.keys():
    #                 raise ValueError(f"Given positive_class '{self.positive_class}' is not present in dataframe.")
    #             else:
    #                 return self.positive_class

    #     return None

    def _get_sensitive_attr_transformer(self, meta: ValueMeta, column_name: str) -> Optional[Transformer]:

        if isinstance(meta, Scale):
            return BinningTransformer(column_name, bins=self.n_bins, duplicates='drop',
                                      include_lowest=True, remove_outliers=0.1)
        elif isinstance(meta, Date):
            return BinningTransformer(column_name, bins=self.n_bins, remove_outliers=None,
                                          duplicates='drop', include_lowest=True)
        else:
            return None

    def _get_dataframe_subset(self, df: pd.DataFrame) -> pd.DataFrame:
        if not all(col in df.columns for col in self._used_columns):
            raise KeyError("Target variable or sensitive attributes not present in DataFrame.")

        df = df[self._used_columns].copy()
        df = df = df[~df[self.target].isna()]

        return df