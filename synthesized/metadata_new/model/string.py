from typing import Dict, List, Optional, Sequence, cast

import numpy as np
import pandas as pd
import rstr

from ..base import DiscreteModel
from ..base.value_meta import NType
from ..exceptions import ModelNotFittedError


class FormattedString(DiscreteModel[str]):
    """A model to sample formatted strings from a regex pattern.

    Attributes:
        name (str): The name of the data column to model.
        categories (Sequence[String]): A list of regex patterns.
        probabilities (Dict[String, float]): A mapping of each of the regex patterns to a probability.

    """

    def __init__(
            self, name: str, categories: Optional[Sequence[str]] = None, nan_freq: Optional[float] = None,
            probabilities: Optional[Dict[NType, float]] = None
    ):
        super().__init__(name=name, categories=categories, nan_freq=nan_freq)
        if self.categories is not None:
            self._extracted = True

        if self.categories is not None and len(self.categories) == 1:
            probabilities = {self.categories[0]: 1.0}

        self.probabilities: Optional[Dict[str, float]] = probabilities
        if self.probabilities is not None or (self.categories is not None and len(self.categories) == 1):
            self._fitted = True

    def probability(self, x: str):
        if not self._fitted:
            raise ModelNotFittedError

        self.probabilities = cast(Dict[str, float], self.probabilities)
        return self.probabilities.get(x, 0.0)

    def sample(self, n: int, produce_nans: bool = False) -> pd.DataFrame:
        if not self._fitted:
            raise ModelNotFittedError

        self.probabilities = cast(Dict[str, float], self.probabilities)
        self.categories = cast(List[str], self.probabilities)

        categories = [*self.categories]
        p = [self.probabilities[c] for c in categories]

        if produce_nans and (self.nan_freq is not None and self.nan_freq > 0):
            p = [i * (1 - self.nan_freq) for i in p]
            p += [self.nan_freq]
            categories.append(np.nan)

        if len(categories) == 1:
            samples = [rstr.xeger(categories[0]) for _ in range(n)]
        else:
            samples = [rstr.xeger(c) if c != 'nan' else np.nan for c in np.random.choice(categories, p=p, size=n)]

        return pd.DataFrame({self.name: samples})
