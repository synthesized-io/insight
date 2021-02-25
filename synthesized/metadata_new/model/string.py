from typing import Optional, Sequence

import numpy as np
import pandas as pd
import rstr

from ..base import DiscreteModel
from ..exceptions import ModelNotFittedError
from ..value import FormattedString


class FormattedStringModel(FormattedString, DiscreteModel[str]):
    """A model to sample formatted strings from a regex pattern.

    Attributes:
        name (str): The name of the data column to model.
        categories (MutableSequence[NType]): A list of the categories.
        probabilities (Dict[NType, float]): A mapping of each of the categories to a probability.
        pattern (str): The regex pattern from which to generate matching strings.

    """

    def __init__(
            self, name: str, categories: Optional[Sequence[str]] = None, nan_freq: Optional[float] = None,
            pattern: str = None
    ):
        super().__init__(name=name, categories=categories, nan_freq=nan_freq, pattern=pattern)

        if nan_freq is not None:
            self._fitted = True

    def sample(self, n: int, produce_nans: bool = False) -> pd.DataFrame:
        if not self._fitted:
            raise ModelNotFittedError

        samples = pd.Series([rstr.xeger(self.pattern) for _ in range(n)], name=self.name)

        if produce_nans and (self.nan_freq is not None and self.nan_freq > 0):
            is_nan = np.random.binomial(1, p=self.nan_freq, size=n) == 1
            samples[is_nan] = np.nan
        return samples.to_frame()

    @classmethod
    def from_meta(cls, meta: FormattedString) -> 'FormattedStringModel':
        return cls(name=meta.name, nan_freq=meta.nan_freq, pattern=meta.pattern)


class SequentialFormattedString(DiscreteModel[str]):
    """A model to sample formatted sequential numeric identifiers with an optional suffix or prefix.

    Attributes:
        name (str): The name of the data column to model.
        categories (MutableSequence[NType]): A list of the categories.
        probabilities (Dict[NType, float]): A mapping of each of the categories to a probability.
        length (int): The length of the numeric identifier. Numbers are zero padded up to this length.
        prefix (str): The prefix of the identifier.
        suffix (str): The suffix of the identifier.

    """

    def __init__(
            self, name: str, categories: Optional[Sequence[str]] = None, nan_freq: Optional[float] = None,
            length: int = None, prefix: str = None, suffix: str = None
    ):
        super().__init__(name=name, categories=categories, nan_freq=nan_freq)
        self.length = length
        self.prefix = prefix
        self.suffix = suffix

        if nan_freq is not None:
            self._fitted = True

    def sample(self, n: int, produce_nans: bool = False) -> pd.DataFrame:
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
