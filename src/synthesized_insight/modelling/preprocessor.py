import logging
from typing import Dict, List, Optional, Sequence, Union, cast

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from ..check import Check, ColumnCheck

logger = logging.getLogger(__name__)


class ModellingPreprocessor:
    def __init__(self, target: Optional[str], check: Check = ColumnCheck()):
        self.target = target
        self.column_encoders: Dict[str, BaseEstimator] = dict()
        self.columns_mapping: Dict[str, List[str]] = dict()
        self.is_fitted: bool = False
        self.check = check
        self.continuous_cols: Union[None, Sequence[str]] = None
        self.categorical_cols: Union[None, Sequence[str]] = None

    def _continuous(self, df) -> Sequence[str]:
        if self.continuous_cols is None:
            self.continuous_cols = [col for col in df.columns if self.check.continuous(df[col])]
        return self.continuous_cols

    def _categorical(self, df) -> Sequence[str]:
        if self.categorical_cols is None:
            self.categorical_cols = [col for col in df.columns if self.check.categorical(df[col])]
        return self.categorical_cols

    def fit(self, df: pd.DataFrame):
        if self.is_fitted:
            logger.info("Preprocessor has already been fitted.")
            return

        for col in df.columns:
            df[col] = self.check.infer_dtype(df[col])

        for cat_col in self._categorical(df):
            logger.debug(f"Preprocessing value '{cat_col}'...")
            column = df[cat_col]
            column = cast(pd.Series, column.fillna('nan').astype(str))
            if self.target and cat_col == self.target:
                x_i = column.to_numpy().reshape(-1, 1)
                encoder = LabelEncoder()
                encoder.fit(x_i)
                c_name_i = [cat_col]

            else:
                x_i = column.to_numpy().reshape(-1, 1)
                encoder = OneHotEncoder(drop='first', sparse=False)
                encoder.fit(x_i)
                c_name_i = cast(List[str],
                                ['{}_{}'.format(cat_col, enc)
                                 for enc in encoder.categories_[0][1:]])

            self.column_encoders[cat_col] = encoder
            self.columns_mapping[cat_col] = c_name_i

        for cont_col in self._continuous(df):
            logger.debug(f"Preprocessing value '{cont_col}'...")
            column = df[cont_col]
            x_i = pd.to_numeric(column, errors="coerce").to_numpy()\
                                                        .reshape(-1, 1)

            n_rows = len(df)
            nan_freq = df[cont_col].isna().sum() / n_rows if n_rows > 0 else 0
            if nan_freq is not None and nan_freq > 0:
                self.column_encoders[cont_col] = Pipeline([('imputer', SimpleImputer()),
                                                           ('scaler', StandardScaler())])
            else:
                self.column_encoders[cont_col] = StandardScaler()

            self.column_encoders[cont_col].fit(x_i)
            self.columns_mapping[cont_col] = [cont_col]

        self.is_fitted = True

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Preprocessor has not been fit yet.\
                              Call 'fit()' before calling 'transform()'")
        df = df.copy()

        xx = []
        c_names: List[str] = []

        for cat_col in self._categorical(df):
            column = df[cat_col]
            column = cast(pd.Series, column.fillna('nan').astype(str))
            encoder = self.column_encoders[cat_col]
            if self.target and cat_col == self.target:
                x_i = encoder.transform(column.values.reshape(-1, 1)).reshape(-1, 1)
                c_name_i = [cat_col]
            else:
                x_i = encoder.transform(column.values.reshape(-1, 1))
                c_name_i = ['{}_{}'.format(cat_col, enc)
                            for enc in encoder.categories_[0][1:]]

            xx.append(x_i)
            c_names.extend(c_name_i)

        for cont_col in self._continuous(df):
            if cont_col not in df.columns:
                continue
            column = df[cont_col]
            x_i = pd.to_numeric(column, errors="coerce").to_numpy()\
                                                        .reshape(-1, 1)
            x_i = self.column_encoders[cont_col].transform(x_i)
            c_name_i = [cont_col]
            xx.append(x_i)
            c_names.extend(c_name_i)

        if len(xx) == 0 or len(c_names) == 0:
            return pd.DataFrame()

        return pd.DataFrame(np.hstack(xx), columns=c_names)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)

    @property
    def processed_columns(self):
        if not self.is_fitted:
            raise ValueError("Preprocessor has not been fit yet.\
                        Call 'fit()' before accessing 'processed_columns'")

        return np.concatenate([self.columns_mapping.values()])

    @classmethod
    def preprocess(cls,
                   df: pd.DataFrame,
                   target: Optional[str]) -> pd.DataFrame:
        preprocessor = cls(target=target)
        return preprocessor.fit_transform(df)
