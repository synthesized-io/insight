import logging
from typing import Dict, List, Optional, Sequence, cast

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from ...metadata.factory import MetaExtractor
from ...model import ContinuousModel, DataFrameModel, DiscreteModel
from ...model.factory import ModelFactory

logger = logging.getLogger(__name__)


class ModellingPreprocessor:
    def __init__(self, target: Optional[str], df_model: Optional[DataFrameModel] = None):
        self.target = target
        self.df_model: Optional[DataFrameModel] = df_model

        self.column_encoders: Dict[str, BaseEstimator] = dict()
        self.columns_mapping: Dict[str, List[str]] = dict()
        self.is_fitted: bool = False

    @property
    def _continuous(self) -> Sequence[ContinuousModel]:
        if self.df_model is None:
            return []
        else:
            return [model for model in self.df_model.values() if isinstance(model, ContinuousModel)]

    @property
    def _categorical(self) -> Sequence[DiscreteModel]:
        if self.df_model is None:
            return []
        else:
            return [model for model in self.df_model.values() if isinstance(model, DiscreteModel)]

    def fit(self, data: pd.DataFrame):
        if self.is_fitted:
            logger.info("Preprocessor has already been fitted.")
            return

        if self.df_model is None:
            df_meta = MetaExtractor.extract(data)
            self.df_model = ModelFactory()(df_meta)

        for v in self._categorical:
            if v.name not in data.columns:
                continue
            logger.debug(f"Preprocessing value '{v.name}'...")
            column = data[v.name]
            column = column.fillna('nan').astype(str)
            if self.target and v.name == self.target:
                x_i = column.to_numpy().reshape(-1, 1)
                encoder = LabelEncoder()
                encoder.fit(x_i)
                c_name_i = [v.name]

            else:
                x_i = column.to_numpy().reshape(-1, 1)
                encoder = OneHotEncoder(drop='first', sparse=False)
                encoder.fit(x_i)
                c_name_i = cast(List[str], ['{}_{}'.format(v.name, enc) for enc in encoder.categories_[0][1:]])

            self.column_encoders[v.name] = encoder
            self.columns_mapping[v.name] = c_name_i

        for w in self._continuous:
            if w.name not in data.columns:
                continue
            logger.debug(f"Preprocessing value '{w.name}'...")
            column = data[w.name]
            x_i = pd.to_numeric(column, errors="coerce").to_numpy().reshape(-1, 1)
            if w.nan_freq is not None and w.nan_freq > 0:
                self.column_encoders[w.name] = Pipeline([('imputer', SimpleImputer()),
                                                         ('scaler', StandardScaler())])
            else:
                self.column_encoders[w.name] = StandardScaler()

            self.column_encoders[w.name].fit(x_i)
            self.columns_mapping[w.name] = [w.name]

        self.is_fitted = True

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Preprocessor has not been fit yet. Call 'fit()' before calling 'transform()'")
        data = data.copy()

        xx = []
        c_names: List[str] = []

        for v in self._categorical:
            if v.name not in data.columns:
                continue
            column = data[v.name]
            column = column.fillna('nan').astype(str)
            encoder = self.column_encoders[v.name]
            if self.target and v.name == self.target:
                x_i = encoder.transform(column.values.reshape(-1, 1)).reshape(-1, 1)
                c_name_i = [v.name]
            else:
                x_i = encoder.transform(column.values.reshape(-1, 1))
                c_name_i = ['{}_{}'.format(v.name, enc) for enc in encoder.categories_[0][1:]]

            xx.append(x_i)
            c_names.extend(c_name_i)

        for w in self._continuous:
            if w.name not in data.columns:
                continue
            column = data[w.name]
            x_i = pd.to_numeric(column, errors="coerce").to_numpy().reshape(-1, 1)
            x_i = self.column_encoders[w.name].transform(x_i)
            c_name_i = [w.name]
            xx.append(x_i)
            c_names.extend(c_name_i)

        if len(xx) == 0 or len(c_names) == 0:
            return pd.DataFrame()

        xx = np.hstack(xx)
        return pd.DataFrame(xx, columns=c_names)

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        self.fit(data)
        return self.transform(data)

    @property
    def processed_columns(self):
        if not self.is_fitted:
            raise ValueError("Preprocessor has not been fit yet. Call 'fit()' before accessing 'processed_columns'")

        return np.concatenate([self.columns_mapping.values()])

    @classmethod
    def preprocess(cls, data: pd.DataFrame, target: Optional[str], df_model: DataFrameModel = None) -> pd.DataFrame:
        preprocessor = cls(target=target, df_model=df_model)
        return preprocessor.fit_transform(data)
