from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .value_meta import ValueMeta

DISTRIBUTIONS = dict(
    beta=np.random.beta,
    gamma=np.random.gamma,
    gumbel=np.random.gumbel,
    normal=np.random.normal,
    uniform=np.random.uniform,
    weibull=np.random.weibull
)


class NumericMeta(ValueMeta):

    def __init__(self, name: str, integer: bool = False, distribution: str = 'uniform',
                 **distribution_kwargs: Dict[str, Any]):
        super().__init__(name=name)

        self.distribution = distribution
        self.distribution_kwargs = distribution_kwargs if distribution_kwargs else dict()
        self.integer = integer

    def specification(self) -> dict:
        spec = super().specification()
        spec.update(distribution=self.distribution, **self.distribution_kwargs)
        return spec

    def extract(self, df: pd.DataFrame) -> None:
        raise NotImplementedError

    def learned_input_columns(self) -> List[str]:
        return []

    def learned_output_columns(self) -> List[str]:
        return []

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(labels=self.name, axis=1)
        return super().preprocess(df=df)

    def postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().postprocess(df=df)
        df[self.name] = DISTRIBUTIONS[self.distribution](size=len(df), **self.distribution_kwargs)

        if self.integer:
            df[self.name] = df[self.name].astype(int)

        return df
