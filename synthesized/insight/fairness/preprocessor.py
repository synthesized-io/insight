from datetime import datetime
import logging
from enum import Enum
from math import log
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from ...config import MetaExtractorConfig

logger = logging.getLogger(__name__)


class VariableType(Enum):
    Binary = 0
    Multinomial = 1
    Continuous = 2


class FairnessPreprocessor:
    def __init__(self, sensitive_attrs: List[str], target: str, n_bins: int = 5,
                 target_n_bins: Optional[int] = 5, drop_dates: bool = True):

        self.sensitive_attrs = sensitive_attrs
        self.target = target
        self.n_bins = n_bins
        self.target_n_bins = target_n_bins
        self.drop_dates = drop_dates

        self.target_variable_type: Optional[VariableType] = None
        self.positive_class: Optional[str] = None
        self.column_bins: Dict[str, List[np.ndarray]] = dict()
        self.column_transformers: Dict[str, Callable[[pd.DataFrame, str], pd.DataFrame]] = dict()
        self.fitted = False

    def fit(self, df: pd.DataFrame, positive_class: Optional[str] = None) -> None:
        if not all(c in df.columns for c in self.sensitive_attrs_and_target):
            raise ValueError("Given DF must contain all sensitive attributes and the target variable.")

        df = df[~df[self.target].isna()].copy()

        other_columns = list(filter(lambda c: c not in self.sensitive_attrs_and_target, df.columns))
        df.drop(other_columns, axis=1, inplace=True)
        if len(df) == 0:
            return

        categorical_threshold = int(max(
            float(MetaExtractorConfig.min_num_unique),
            MetaExtractorConfig.categorical_threshold_log_multiplier * log(len(df))
        ))

        for col in self.sensitive_attrs_and_target:
            df[col] = self.transform_dtype(df[col])

            categorical_threshold = categorical_threshold if categorical_threshold is not None else self.n_bins
            num_unique = df[col].nunique()

            # If it's numeric, bin it
            if df[col].dtype.kind in ('i', 'u', 'f'):
                if col != self.target:
                    if num_unique > categorical_threshold:
                        self.compute_column_bins(df[col], n_bins=self.n_bins)
                        self.column_transformers[col] = self.bin_column
                    else:
                        self.column_transformers[col] = self.categorical_transformer

                else:  # col == target
                    if num_unique > categorical_threshold:
                        if self.target_n_bins is not None:
                            self.compute_column_bins(df[col], n_bins=self.target_n_bins)
                            self.column_transformers[col] = self.bin_column
                            self.target_variable_type = VariableType.Binary if num_unique <= 2 \
                                else VariableType.Multinomial
                        else:
                            self.column_transformers[col] = self.do_nothing
                            self.target_variable_type = VariableType.Continuous
                    else:
                        self.column_transformers[col] = self.categorical_transformer
                        self.target_variable_type = VariableType.Binary if num_unique <= 2 else VariableType.Multinomial

            # If it's datetime, bin it
            elif df[col].dtype.kind == 'M':
                if self.drop_dates:
                    self.column_transformers[col] = self.drop_column
                    self.sensitive_attrs.remove(col)
                    logging.info(f"Sensitive attribute '{col}' dropped as it is a date value and 'drop_dates=True'.")
                else:
                    self.column_transformers[col] = self.bin_date_column

                if col == self.target:
                    raise TypeError(f"Datetime target columns not supported")

            # If it's a sampling value, discard it
            elif num_unique > np.sqrt(len(df)):
                if col == self.target:
                    raise TypeError(f"Target column ")

                else:
                    self.column_transformers[col] = self.drop_column
                    self.sensitive_attrs.remove(col)
                    logging.info(f"Sensitive attribute '{col}' dropped as it is a sampled value.")

            # Pure categorical
            else:
                self.column_transformers[col] = self.categorical_transformer
                if col == self.target:
                    self.target_variable_type = VariableType.Binary if df[col].nunique() <= 2 \
                        else VariableType.Multinomial

        self.positive_class = self.get_positive_class(df, positive_class=positive_class)
        self.fitted = True

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

        for col in self.column_transformers.keys():
            transformer = self.column_transformers[col]
            df = transformer(df, col)

        return df

    def get_positive_class(self, df: pd.DataFrame, positive_class: Optional[str]) -> Optional[str]:
        # Only set positive class for binary/multinomial, even if given.
        transformer = self.column_transformers[self.target]
        target_column = transformer(df[[self.target]].copy(), self.target)[self.target]

        if self.target_variable_type in (VariableType.Binary, VariableType.Multinomial):
            target_vc = target_column.value_counts(normalize=True)
            if len(target_vc) <= 2:
                if positive_class is None:
                    # If target class is not given, we'll use minority class as usually it is the target.
                    return str(target_vc.idxmin())
                elif positive_class not in target_vc.keys():
                    raise ValueError(f"Given positive_class '{positive_class}' is not present in dataframe.")
                else:
                    return positive_class

        return None

    @property
    def sensitive_attrs_and_target(self) -> List[str]:
        return list(np.concatenate((self.sensitive_attrs, [self.target])))

    @staticmethod
    def transform_dtype(column: pd.Series) -> pd.Series:
        # Try to convert it to numeric if it isn't
        if column.dtype.kind not in ('i', 'u', 'f'):
            n_nans = column.isna().sum()
            col_num = pd.to_numeric(column, errors='coerce')
            if col_num.isna().sum() == n_nans:
                return col_num

        # Try to convert it to date
        if column.dtype.kind == 'O':
            n_nans = column.isna().sum()
            try:
                col_date = pd.to_datetime(column, errors='coerce')
            except TypeError:  # Argument 'date_string' has incorrect type (expected str, got numpy.str_)
                col_date = pd.to_datetime(column.astype(str), errors='coerce')

            if col_date.isna().sum() == n_nans:
                return col_date

        return column

    def compute_column_bins(self, column: pd.Series, n_bins: Optional[int] = None,
                            remove_outliers: float = 0.1) -> None:
        name = str(column.name)
        if name in self.column_bins:
            raise ValueError(f"Bins for column '{name}' have already been computed.")

        column_clean = column.copy().dropna()
        percentiles = [remove_outliers * 100. / 2, 100 - remove_outliers * 100. / 2]
        start, end = np.percentile(column_clean, percentiles)

        if start == end:
            start, end = min(column_clean), max(column_clean)
        column_clean = column_clean[(start <= column_clean) & (column_clean <= end)]

        _, bins = pd.cut(column_clean, bins=n_bins, retbins=True, duplicates='drop', include_lowest=True)
        assert isinstance(bins, np.ndarray)
        bins[0], bins[-1] = column.min(), column.max()

        self.column_bins[str(column.name)] = bins

        if name == self.target:
            self.target_variable_type = VariableType.Continuous

    def bin_column(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        if name not in self.column_bins:
            raise ValueError(f"Bins for column '{name}' haven't been computed yet, call compute_column_bins() first.")

        df[name] = pd.cut(df[name], bins=self.column_bins[name], duplicates='drop', include_lowest=True).astype(str)
        return df

    @staticmethod
    def drop_column(df: pd.DataFrame, name: str) -> pd.DataFrame:
        return df.drop(name, axis=1, errors='ignore')

    @staticmethod
    def categorical_transformer(df: pd.DataFrame, name: str) -> pd.DataFrame:
        df[name] = df[name].astype(str).fillna('nan')
        return df

    @staticmethod
    def do_nothing(df: pd.DataFrame, name: str) -> pd.DataFrame:
        return df

    def bin_date_column(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        date_format: str = "%Y-%m-%d"
        column = df[name]

        assert column.dtype.kind == "M"
        column, edges_num = pd.cut(column, bins=self.n_bins, retbins=True, duplicates='drop', include_lowest=True)

        edges_str = [datetime.strftime(pd.to_datetime(d), date_format) for d in edges_num]
        edges = {pd.Interval(edges_num[i], edges_num[i + 1]): "{} to {}".format(edges_str[i], edges_str[i + 1])
                 for i in range(len(edges_num) - 1)}

        df[name] = column.map(edges).astype(str)
        return df
