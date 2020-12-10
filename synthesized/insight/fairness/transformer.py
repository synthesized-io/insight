from datetime import datetime
import logging
from enum import Enum
from math import log
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from ...transformer import Transformer, DTypeTransformer, BinningTransformer, DropColumnTransformer
from ...transformer.base import TransformerType
from ...config import MetaExtractorConfig

logger = logging.getLogger(__name__)


class VariableType(Enum):
    Binary = 0
    Multinomial = 1
    Continuous = 2


class FairnessTransformer(Transformer):
    """
    Fairness transformer

    Attributes:
        name (str) : the data frame column to transform.
    """

    def __init__(self, sensitive_attrs: List[str], target: str, n_bins: int = 5,
                 target_n_bins: Optional[int] = 5, positive_class: Optional[str] = None,
                 drop_dates: bool = True):

        self.sensitive_attrs = sensitive_attrs
        self.target = target
        self.n_bins = n_bins
        self.target_n_bins = target_n_bins
        self.positive_class = positive_class
        self.drop_dates = drop_dates

        self.target_variable_type: Optional[VariableType] = None

        super().__init__(name="fairness_transformer")

    def fit(self, df: pd.DataFrame) -> 'FairnessTransformer':
        if not all(c in df.columns for c in self.sensitive_attrs_and_target):
            raise ValueError("Given DF must contain all sensitive attributes and the target variable.")

        df = df[~df[self.target].isna()].copy()

        other_columns = list(filter(lambda c: c not in self.sensitive_attrs_and_target, df.columns))
        df.drop(other_columns, axis=1, inplace=True)
        if len(df) == 0:
            return self

        categorical_threshold = int(max(
            float(MetaExtractorConfig.min_num_unique),
            MetaExtractorConfig.categorical_threshold_log_multiplier * log(len(df))
        ))

        for col in self.sensitive_attrs_and_target:

            dtype_transformer = DTypeTransformer(col)
            df = dtype_transformer.fit_transform(df)

            categorical_threshold = categorical_threshold if categorical_threshold is not None else self.n_bins
            num_unique = df[col].nunique()

            transformer: Optional['Transformer'] = None
            # If it's numeric, bin it
            if df[col].dtype.kind in ('i', 'u', 'f'):
                if col != self.target:
                    if num_unique > categorical_threshold:
                        transformer = BinningTransformer(col, bins=self.n_bins, duplicates='drop', include_lowest=True)

                else:  # col == target
                    if num_unique > categorical_threshold:
                        if self.target_n_bins is not None:
                            transformer = BinningTransformer(col, bins=self.target_n_bins, duplicates='drop',
                                                             include_lowest=True)

                            self.target_variable_type = VariableType.Binary if num_unique <= 2 else \
                                VariableType.Multinomial
                        else:
                            self.target_variable_type = VariableType.Continuous
                    else:
                        self.target_variable_type = VariableType.Binary if num_unique <= 2 else VariableType.Multinomial

            # If it's datetime, bin it
            elif df[col].dtype.kind == 'M':
                print(f"{col} is date")
                if self.drop_dates:
                    transformer = DropColumnTransformer(col)
                    self.sensitive_attrs.remove(col)
                    logging.info(f"Sensitive attribute '{col}' dropped as it is a date value and 'drop_dates=True'.")
                else:
                    transformer = BinningTransformer(col, bins=self.n_bins, remove_outliers=None,
                                                     duplicates='drop', include_lowest=True)

                if col == self.target:
                    raise TypeError("Datetime target columns not supported")

            # If it's a sampling value, discard it
            elif num_unique > np.sqrt(len(df)):
                if col == self.target:
                    raise TypeError("Target column has too many unique non-numerical values to compute fairness.")

                else:
                    transformer = DropColumnTransformer(col)
                    self.sensitive_attrs.remove(col)
                    logging.info(f"Sensitive attribute '{col}' dropped as it is a sampled value.")

            # Pure categorical
            else:
                # transformer = DTypeTransformer(col, out_dtype='str')
                if col == self.target:
                    self.target_variable_type = VariableType.Binary if df[col].nunique() <= 2 \
                        else VariableType.Multinomial

            self.append(dtype_transformer)
            if transformer is not None:
                self.append(transformer.fit(df))

            to_str_transformer = DTypeTransformer(col, out_dtype='str')
            self.append(to_str_transformer.fit(df))

        self.positive_class = self.get_positive_class(df)
        self.fitted = True

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise ValueError("Preprocessor hasn't been fitted yet, please call fit().")

        if not all(c in df.columns for c in self.sensitive_attrs_and_target):
            raise ValueError("Given DF must contain all sensitive attributes and the target variable.")

        df = df[~df[self.target].isna()].copy()

        other_columns = list(filter(lambda c: c not in self.sensitive_attrs_and_target, df.columns))
        df.drop(other_columns, axis=1, inplace=True)
        if len(df) == 0:
            return df

        for transformer in self:
            df = transformer(df)

        return df

    def get_positive_class(self, df: pd.DataFrame) -> Optional[str]:
        # Only set positive class for binary/multinomial, even if given.
        df_target = df[[self.target]].copy()
        for transformer in [t for t in self if t.name == self.target]:
            df_target = transformer.transform(df_target)

        target_column = df_target[self.target]

        if self.target_variable_type in (VariableType.Binary, VariableType.Multinomial):
            target_vc = target_column.value_counts(normalize=True)
            if len(target_vc) <= 2:
                if self.positive_class is None:
                    # If target class is not given, we'll use minority class as usually it is the target.
                    return str(target_vc.idxmin())
                elif self.positive_class not in target_vc.keys():
                    raise ValueError(f"Given positive_class '{self.positive_class}' is not present in dataframe.")
                else:
                    return self.positive_class

        return None

    @property
    def sensitive_attrs_and_target(self) -> List[str]:
        return list(np.concatenate((self.sensitive_attrs, [self.target])))
