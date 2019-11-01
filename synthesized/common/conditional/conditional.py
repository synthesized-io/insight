import re
from abc import ABC
from collections import Counter
from itertools import product
from typing import Any, Dict, Tuple, Union, Callable, List

import numpy as np
import pandas as pd

from ...common.values.continuous import ContinuousValue
from ...synthesizer import Synthesizer


class ConditionalSampler(Synthesizer):
    """
    Samples from the synthesizer conditionally on explicitly defined marginals of some columns.
    """

    def __init__(self,
                 synthesizer: Synthesizer,
                 *explicit_marginals: Tuple[str, Dict[Any, float]],
                 min_sampled_ratio: float = 0.001):
        """Create ConditionalSampler.

        Args:
            synthesizer: An underlying synthesizer
            *explicit_marginals: A dict of desired marginal distributions per column.
                Distributions defined as density per category or bin. The result will be sampled
                from the synthesizer conditionally on these marginals.
            min_sampled_ratio: Stop synthesis if ratio of successfully sampled records is less than given value.
        """
        self.synthesizer = synthesizer

        # For simplicity let's store distributions in a dict where key is a column:
        self.explicit_marginals: Dict[str, Dict[Any, float]] = {}
        for col, cond in explicit_marginals:
            assert np.isclose(sum(cond.values()), 1.0)
            self.explicit_marginals[col] = cond
        self.conditional_columns: List[str] = []

        # Let's compute cartesian product of all probs for each column
        # to get probs for the joined distribution:
        category_probs = []
        for column, distr in explicit_marginals:
            self.conditional_columns.append(column)
            category_probs.append([(category, prob) for category, prob in distr.items()])
        category_combinations = product(*category_probs)
        rows = [
            tuple(zip(*comb))
            for comb in category_combinations
        ]
        self.joined_marginal_probs = {row[0]: np.product(row[1]) for row in rows}
        self.min_sampled_ratio = min_sampled_ratio

    def learn(self, num_iterations: int, df_train: pd.DataFrame,
              callback: Callable[[object, int, dict], bool] = Synthesizer.logging, callback_freq: int = 0) -> None:
        self.synthesizer.learn(
            num_iterations=num_iterations, df_train=df_train, callback=callback, callback_freq=callback_freq
        )

    def synthesize(self,
                   num_rows: int,
                   conditions: Union[dict, pd.DataFrame] = None,
                   progress_callback: Callable[[int], None] = None):
        # For the sake of performance we will not really sample from "condition" distribution,
        # but will rather sample directly from synthesizer and filter records so they distribution is conditional

        # Counters represent hom many instances of each condition we need to sample
        marginal_counts = Counter({c: int(round(p * num_rows)) for c, p in self.joined_marginal_probs.items()})

        # Let's adjust counts so they sum up to `num_rows`:
        any_key = list(marginal_counts.keys())[0]
        marginal_counts[any_key] += num_rows - sum(marginal_counts.values())

        # The result is a list of result arrays
        result = []

        sampled_ratio = 1.0
        all_columns = None
        while sum(marginal_counts.values()) > 0 and sampled_ratio >= self.min_sampled_ratio:
            n_missing = sum(marginal_counts.values())

            # Estimate how many rows we need so after filtering we have enough:
            n_prefetch = round(n_missing / sampled_ratio * 1.1)
            n_prefetch = min(n_prefetch, int(1e6))

            # Synthesis:
            df_synthesized = self.synthesizer.synthesize(num_rows=n_prefetch, conditions=conditions)

            # In order to filter our data frame we need keys that we will look up in counts:
            df_key = df_synthesized[self.conditional_columns]
            df_key = self._map_continuous_columns(df_key)

            all_columns = df_synthesized.columns
            n_added = 0
            for key_row, row in zip(df_key.to_numpy(), df_synthesized.to_numpy()):
                key = tuple(key_row)
                # If counter for the instance is positive let's emit the current row:
                if marginal_counts[key] > 0:
                    result.append(row)
                    n_added += 1
                    marginal_counts[key] -= 1
            if n_added == 0:
                # In case if we couldn't sample anything this time:
                sampled_ratio = 1.0 / n_prefetch
            else:
                sampled_ratio = float(n_added) / n_prefetch
        return pd.DataFrame.from_records(result, columns=all_columns)

    def _map_continuous_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Looks for continuous columns and map values into bins that are defined in `self.conditions`.

        Args:
            df: Input data frame.

        Returns:
            Result data frame.

        """
        df = df.copy()

        mapping = {}
        continuos_columns = {v.name for v in self.synthesizer.values if isinstance(v, ContinuousValue)}  # type: ignore
        for col in continuos_columns:
            if col in self.explicit_marginals:
                intervals = []
                for str_interval in self.explicit_marginals[col].keys():
                    interval = FloatInterval.parse(str_interval)
                    intervals.append(interval)
                mapping[col] = intervals

        for col in self.conditional_columns:
            if col in continuos_columns:
                def map_value(value: float):
                    intervals = mapping[col]
                    for interval in intervals:
                        if interval.is_in(value):
                            return str(interval)
                df[col] = df[col].apply(map_value)

        return df


class FloatEndpoint(ABC):
    def __init__(self, value: float):
        self.value = value

    def to_str(self, is_left: bool) -> str:
        pass

    def as_left_in(self, value: float) -> bool:
        pass

    def as_right_in(self, value: float) -> bool:
        pass


class Inclusive(FloatEndpoint):
    def __init__(self, value: float):
        super().__init__(value)

    def as_left_in(self, value: float) -> bool:
        return value >= self.value

    def as_right_in(self, value: float) -> bool:
        return value <= self.value

    def to_str(self, is_left: bool) -> str:
        if is_left:
            return '[{}'.format(self.value)
        else:
            return '{}]'.format(self.value)

    def __eq__(self, other):
        return self.value == other.value

    def __repr__(self):
        return "Inclusive({})".format(self.value)


class Exclusive(FloatEndpoint):
    def __init__(self, value: float):
        super().__init__(value)

    def as_left_in(self, value: float) -> bool:
        return value > self.value

    def as_right_in(self, value: float) -> bool:
        return value < self.value

    def to_str(self, is_left: bool) -> str:
        if is_left:
            return '({}'.format(self.value)
        else:
            return '{})'.format(self.value)

    def __eq__(self, other):
        return self.value == other.value

    def __repr__(self):
        return "Exclusive({})".format(self.value)


class FloatInterval:
    """Models an interval of float values."""

    RE = re.compile(r'([\[\(])(\S+\.\S+),\s(\S+\.\S+)([\]\)])')

    def __init__(self, left: FloatEndpoint, right: FloatEndpoint):
        assert left.value < right.value
        self.left = left
        self.right = right

    def is_in(self, value: float) -> bool:
        return self.left.as_left_in(value) and self.right.as_right_in(value)

    def __eq__(self, other):
        return self.left == other.left and self.right == other.right

    def __repr__(self):
        return 'FloatInterval({}, {})'.format(self.left, self.right)

    def __str__(self) -> str:
        return '{left}, {right}'.format(left=self.left.to_str(is_left=True), right=self.right.to_str(is_left=False))

    @classmethod
    def parse(cls, s: str):
        m = FloatInterval.RE.match(s)
        assert m

        left_bracket, left_s, right_s, right_bracket = m.groups()
        left, right = float(left_s), float(right_s)

        if left_bracket == '[':
            left_endpoint: FloatEndpoint = Inclusive(left)
        elif left_bracket == '(':
            left_endpoint = Exclusive(left)
        else:
            assert False

        if right_bracket == ']':
            right_endpoint: FloatEndpoint = Inclusive(right)
        elif right_bracket == ')':
            right_endpoint = Exclusive(right)
        else:
            assert False

        return FloatInterval(left_endpoint, right_endpoint)
