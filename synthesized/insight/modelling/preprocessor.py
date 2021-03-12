import logging
from typing import Dict, List, Optional, cast

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from ...metadata_new import Affine, DataFrameMeta
from ...metadata_new.factory import MetaExtractor
from ...model import ContinuousModel, DataFrameModel, DiscreteModel
from ...model.factory import ModelFactory

logger = logging.getLogger(__name__)


class ModellingPreprocessor:
    def __init__(self, target: Optional[str], df: pd.DataFrame = None, dp: DataFrameMeta = None, models: DataFrameModel = None):
        self.target = target

        if dp is None and df is not None:
            self.dp: Optional[DataFrameMeta] = MetaExtractor.extract(df)
        elif dp is not None:
            self.dp = dp
        else:
            self.dp = None

        if models is None and self.dp is not None:
            self.models: Optional[DataFrameModel] = ModelFactory()(self.dp)
        else:
            self.models = models

        self.column_encoders: Dict[str, BaseEstimator] = dict()
        self.columns_mapping: Dict[str, List[str]] = dict()

        self.is_fitted: bool = False

    def fit(self, data: pd.DataFrame):
        if self.is_fitted:
            logger.info("Preprocessor has already been fitted.")
            return

        if self.dp is None:
            self.dp = MetaExtractor.extract(data)

        if self.models is None:
            self.models = ModelFactory()(self.dp)

        categorical, continuous = [], []
        for model in self.models.values():
            if isinstance(model, ContinuousModel):
                continuous.append(model.name)
            if isinstance(model, DiscreteModel):
                categorical.append(model.name)

        for v in self.dp.values():
            logger.debug(f"Preprocessing value '{v.name}'...")
            column = data[v.name]

            if v.name in categorical:
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
                    c_name_i = ['{}_{}'.format(v.name, enc) for enc in encoder.categories_[0][1:]]

                self.column_encoders[v.name] = encoder

            elif v.name in continuous:
                x_i = pd.to_numeric(column, errors="coerce").to_numpy().reshape(-1, 1)
                v = cast(Affine, v)
                if v.nan_freq is not None and v.nan_freq > 0:
                    self.column_encoders[v.name] = Pipeline([('imputer', SimpleImputer()),
                                                             ('scaler', StandardScaler())])
                else:
                    self.column_encoders[v.name] = StandardScaler()
                self.column_encoders[v.name].fit(x_i)

                c_name_i = [v.name]

            else:
                c_name_i = []
                logger.debug(f"Ignoring column {v.name} (type {v.__class__.__name__})")

            self.columns_mapping[v.name] = c_name_i

        self.is_fitted = True

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Preprocessor has not been fit yet. Call 'fit()' before calling 'transform()'")
        assert self.models is not None
        data = data.copy()

        xx = []
        c_names: List[str] = []

        for col, v in self.models.items():
            column = data[col]
            x_i = None
            c_name_i = None

            if isinstance(v, DiscreteModel):
                column = column.fillna('nan').astype(str)
                encoder = self.column_encoders[v.name]
                if self.target and v.name == self.target:
                    x_i = encoder.transform(column.values.reshape(-1, 1)).reshape(-1, 1)
                    c_name_i = [v.name]
                else:
                    x_i = encoder.transform(column.values.reshape(-1, 1))
                    c_name_i = ['{}_{}'.format(v.name, enc) for enc in encoder.categories_[0][1:]]

            elif isinstance(v, ContinuousModel):
                x_i = pd.to_numeric(column, errors="coerce").to_numpy().reshape(-1, 1)
                x_i = self.column_encoders[v.name].transform(x_i)
                c_name_i = [v.name]

            else:
                logger.debug(f"Ignoring column {col} (type {v.__class__.__name__})")

            if x_i is not None and c_name_i is not None:
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
    def preprocess(cls, data: pd.DataFrame, target: Optional[str], dp: DataFrameMeta = None) -> pd.DataFrame:
        preprocessor = cls(target=target, dp=dp)
        return preprocessor.fit_transform(data)
