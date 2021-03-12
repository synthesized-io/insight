from typing import Optional

import numpy as np
import pandas as pd
import rstr

from ..base import DiscreteModel
from ..exceptions import ModelNotFittedError
from ...metadata_new.value import FormattedString, String


class FormattedStringModel(DiscreteModel[FormattedString, str]):
    """A model to sample formatted strings from a regex pattern.

    Attributes:
        name (str): The name of the data column to model.
        categories (MutableSequence[NType]): A list of the categories.
        probabilities (Dict[NType, float]): A mapping of each of the categories to a probability.
        pattern (str): The regex pattern from which to generate matching strings.

    """

    def __init__(self, meta: FormattedString):
        super().__init__(meta=meta)

        if self.nan_freq is not None:
            self._fitted = True

    @property
    def pattern(self) -> Optional[str]:
        return self._meta.pattern

    def sample(self, n: int, produce_nans: bool = False, conditions: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if not self._fitted:
            raise ModelNotFittedError

        samples = pd.Series([rstr.xeger(self.pattern) for _ in range(n)], name=self.name)

        if produce_nans and (self.nan_freq is not None and self.nan_freq > 0):
            is_nan = np.random.binomial(1, p=self.nan_freq, size=n) == 1
            samples[is_nan] = np.nan

        return samples.to_frame()


class SequentialFormattedString(DiscreteModel[String, str]):
    """A model to sample formatted sequential numeric identifiers with an optional suffix or prefix.

    Attributes:
        name (str): The name of the data column to model.
        categories (MutableSequence[NType]): A list of the categories.
        probabilities (Dict[NType, float]): A mapping of each of the categories to a probability.
        length (int): The length of the numeric identifier. Numbers are zero padded up to this length.
        prefix (str): The prefix of the identifier.
        suffix (str): The suffix of the identifier.

    """

    def __init__(self, meta: String, length: int = None, prefix: str = None, suffix: str = None):
        super().__init__(meta=meta)
        self.length = length
        self.prefix = prefix
        self.suffix = suffix

        if self._meta.nan_freq is not None:
            self._fitted = True

    def sample(self, n: int, produce_nans: bool = False, conditions: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if not self._fitted:
            raise ModelNotFittedError

        samples = pd.Series(np.arange(0, n), name=self.name, dtype=str)
        samples = samples.str.zfill(self.length)

        if self.prefix is not None:
            samples = self.prefix + samples

        if self.suffix is not None:
            samples = samples + self.suffix

        if produce_nans and (self.nan_freq is not None and self.nan_freq > 0):
            is_nan = np.random.binomial(1, p=self.nan_freq, size=n) == 1
            samples[is_nan] = np.nan

        return samples.to_frame()
