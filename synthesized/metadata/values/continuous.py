from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import gamma, gilbrat, gumbel_r, lognorm, norm, uniform, weibull_min
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from tensorflow_probability import distributions as tfd

from .value_meta import ValueMeta

DISTRIBUTIONS = dict(
    # beta=(beta, tfd.Beta),
    gamma=(gamma, tfd.Gamma),
    gilbrat=(gilbrat, None),
    gumbel=(gumbel_r, tfd.Gumbel),
    log_normal=(lognorm, tfd.LogNormal),
    uniform=(uniform, tfd.Uniform),
    weibull=(weibull_min, None)
)


class ContinuousMeta(ValueMeta):

    def __init__(
        self, name: str,
        # Scenario
        integer: bool = None, is_float: bool = True, positive: bool = None, nonnegative: bool = None,
        use_quantile_transformation: bool = True, distribution: str = None, distribution_params: Tuple[Any, ...] = None,
        transformer_n_quantiles: int = 1000, transformer_noise: Optional[float] = 1e-7
    ):
        super().__init__(name=name)

        self.integer = integer
        self.is_float = is_float
        self.positive = positive
        self.nonnegative = nonnegative

        assert distribution is None or distribution == 'normal' or distribution in DISTRIBUTIONS
        self.distribution = distribution
        self.distribution_params = distribution_params

        # transformer is fitted in `extract`
        self.use_quantile_transformation = use_quantile_transformation
        self.transformer_n_quantiles = transformer_n_quantiles
        self.transformer_noise = transformer_noise
        self.transformer: Optional[Union[QuantileTransformer, StandardScaler]] = None

        self.pd_types: Tuple[str, ...] = ('f', 'i')

    @staticmethod
    def pd_cast(col: pd.Series) -> pd.Series:
        return pd.to_numeric(col, errors='coerce', downcast='integer')

    def specification(self) -> Dict[str, Any]:
        spec = super().specification()
        spec.update(
            integer=self.integer, positive=self.positive,
            nonnegative=self.nonnegative
        )
        return spec

    def extract(self, df: pd.DataFrame) -> None:
        super().extract(df=df)
        column = df.loc[:, self.name]

        if column.dtype.kind not in self.pd_types:
            column = self.pd_cast(column)

        self.is_float = (column.dtype.kind == 'f')

        if self.integer is None:
            self.integer = (column.dtype.kind == 'i') or column.apply(lambda x: x.is_integer()).all()
        elif self.integer and column.dtype.kind not in ['i', 'u']:
            raise NotImplementedError

        column = column.astype(dtype='float32')
        assert not column.isna().any()
        assert (column != float('inf')).all() and (column != float('-inf')).all()

        if self.positive is None:
            self.positive = (column > 0.0).all()
        elif self.positive and (column <= 0.0).any():
            raise NotImplementedError

        if self.nonnegative is None:
            self.nonnegative = (column >= 0.0).all()
        elif self.nonnegative and (column < 0.0).any():
            raise NotImplementedError

        if self.transformer_noise:
            column += np.random.normal(scale=self.transformer_noise, size=len(column))

        if self.use_quantile_transformation:
            self.transformer = QuantileTransformer(n_quantiles=min(self.transformer_n_quantiles, len(column)),
                                                   output_distribution='normal')
        else:
            self.transformer = StandardScaler()
        self.transformer.fit(column.to_numpy().reshape(-1, 1))

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO: mb removal makes learning more stable (?), an investigation required
        # df = ContinuousValue.remove_outliers(df, self.name, REMOVE_OUTLIERS_PCT)

        if df.loc[:, self.name].dtype.kind not in self.pd_types:
            df.loc[:, self.name] = pd.to_numeric(df.loc[:, self.name], errors='coerce', downcast='integer')

        assert not df.loc[:, self.name].isna().any()
        assert (df.loc[:, self.name] != float('inf')).all() and (df.loc[:, self.name] != float('-inf')).all()

        if self.nonnegative and not self.positive:
            df.loc[:, self.name] = np.maximum(df.loc[:, self.name], 0.001)

        if self.distribution == 'normal':
            assert self.distribution_params is not None
            mean, stddev = self.distribution_params
            df.loc[:, self.name] = (df.loc[:, self.name] - mean) / stddev

        elif self.distribution is not None:
            df.loc[:, self.name] = norm.ppf(
                DISTRIBUTIONS[self.distribution][0].cdf(df.loc[:, self.name], *self.distribution_params)
            )
            df = df[(df.loc[:, self.name] != float('inf')) & (df.loc[:, self.name] != float('-inf'))]
        elif self.transformer:
            if self.transformer_noise:
                df.loc[:, self.name] += np.random.normal(scale=self.transformer_noise, size=len(df.loc[:, self.name]))
            df.loc[:, self.name] = self.transformer.transform(df.loc[:, self.name].values.reshape(-1, 1))

        assert not df.loc[:, self.name].isna().any()
        assert (df.loc[:, self.name] != float('inf')).all() and (df.loc[:, self.name] != float('-inf')).all()

        df.loc[:, self.name] = df.loc[:, self.name].astype(np.float32)
        return super().preprocess(df=df)

    def postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().postprocess(df=df)
        if self.distribution == 'normal':
            assert self.distribution_params is not None
            mean, stddev = self.distribution_params
            df.loc[:, self.name] = df.loc[:, self.name] * stddev + mean
        elif self.distribution is not None:
            df.loc[:, self.name] = DISTRIBUTIONS[self.distribution][0].ppf(
                norm.cdf(df.loc[:, self.name]), *self.distribution_params
            )
        elif self.transformer:
            df.loc[:, self.name] = self.transformer.inverse_transform(df.loc[:, self.name].values.reshape(-1, 1))

        if self.nonnegative:
            df.loc[(df.loc[:, self.name] < 0.001), self.name] = 0

        assert not df.loc[:, self.name].isna().any()
        assert (df.loc[:, self.name] != float('inf')).all() and (df.loc[:, self.name] != float('-inf')).all()

        if self.integer:
            df.loc[:, self.name] = df.loc[:, self.name].astype(dtype='int64')

        if self.is_float and df.loc[:, self.name].dtype != 'float32':
            df.loc[:, self.name] = df.loc[:, self.name].astype(dtype='float32')
        self.set_dtypes(df)
        return df
