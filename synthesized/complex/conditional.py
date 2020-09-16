import gc
import re
from abc import ABC
from collections import Counter
from itertools import product
from typing import Any, Dict, Tuple, Union, Callable, List, Optional

import numpy as np
import pandas as pd
import tensorflow as tf

from ..common.synthesizer import Synthesizer
from ..common.values import ContinuousValue, CategoricalValue, NanValue, Value
from ..metadata import ValueMeta


class ConditionalSampler(Synthesizer):
    """Samples from the synthesizer conditionally on explicitly defined marginals of some columns.

    Example:
        >>> cond = ConditionalSampler(synthesizer)
        >>> cond.synthesize(num_rows=100, explicit_marginals={'SeriousDlqin2yrs': {'0': 0.3, '1': 0.7},
        >>>                                                   'age': {'[0.0, 50.0)': 0.5, '[50.0, 100.0)': 0.5}}))
    """

    def __init__(self,
                 synthesizer: Synthesizer,
                 min_sampled_ratio: float = 0.001,
                 synthesis_batch_size: Optional[int] = 16384):
        """Create ConditionalSampler.

        Args:
            synthesizer: An underlying synthesizer
            min_sampled_ratio: Stop synthesis if ratio of successfully sampled records is less than given value.
            synthesis_batch_size: Synthesis batch size
        """
        super().__init__(name='conditional')
        self.synthesizer = synthesizer
        self.global_step = synthesizer.global_step
        self.logdir = synthesizer.logdir
        self.loss_history = synthesizer.loss_history
        self.writer = synthesizer.writer

        self.min_sampled_ratio = min_sampled_ratio
        self.synthesis_batch_size = synthesis_batch_size

        self.all_columns: List[str] = self.synthesizer.value_factory.columns
        self.continuous_columns = {v.name for v in self.synthesizer.get_values()
                                   if (isinstance(v, ContinuousValue) or isinstance(v, NanValue))}

    def learn(self, df_train: pd.DataFrame, num_iterations: Optional[int],
              callback: Callable[[object, int, dict], bool] = None, callback_freq: int = 0) -> None:
        self.synthesizer.learn(
            num_iterations=num_iterations, df_train=df_train, callback=callback, callback_freq=callback_freq
        )

    def synthesize(self,
                   num_rows: int,
                   conditions: Union[dict, pd.DataFrame] = None,
                   produce_nans: bool = False,
                   progress_callback: Callable[[int], None] = None,
                   explicit_marginals: Dict[str, Dict[str, float]] = None) -> pd.DataFrame:
        """Generate the given number of new data rows according to the ConditionalSynthesizer's explicit marginals.

        Args:
            num_rows: The number of rows to generate.
            conditions: The condition values for the generated rows.
            produce_nans: Whether to produce NaNs.
            progress_callback: Progress bar callback.
            explicit_marginals: A dict of desired marginal distributions per column.
                Distributions defined as density per category or bin. The result will be sampled
                from the synthesizer conditionally on these marginals.

        Returns:
            The generated data.

        """

        if progress_callback is not None:
            progress_callback(0)

        if explicit_marginals is None or len(explicit_marginals) == 0:
            return self.synthesizer.synthesize(num_rows, conditions=conditions, produce_nans=produce_nans,
                                               progress_callback=progress_callback)

        # For the sake of performance we will not really sample from "condition" distribution,
        # but will rather sample directly from synthesizer and filter records so they distribution is conditional
        self._validate_explicit_marginals(explicit_marginals)
        marginal_counts = self.get_joined_marginal_counts(explicit_marginals, num_rows)

        # Let's adjust counts so they sum up to `num_rows`:
        any_key = list(marginal_counts.keys())[0]
        marginal_counts[any_key] += num_rows - sum(marginal_counts.values())

        marginals_keys = {k: list(v.keys()) for k, v in explicit_marginals.items()}

        return self.synthesize_from_joined_counts(marginal_counts, marginals_keys,
                                                  conditions=conditions, progress_callback=progress_callback)

    def synthesize_from_joined_counts(
            self,
            marginal_counts: Dict[Tuple, int],
            marginals_keys: Dict[str, List[str]],
            conditions: Union[dict, pd.DataFrame] = None,
            progress_callback: Callable[[int], None] = None,
    ) -> pd.DataFrame:
        """Given joint counts, synthesize dataset."""

        marginal_counts = marginal_counts.copy()
        num_rows = sum(marginal_counts.values())

        # The result is a list of result arrays
        result = []

        sampled_ratio = 1.01
        while sum(marginal_counts.values()) > 0 and sampled_ratio >= self.min_sampled_ratio:
            n_missing = sum(marginal_counts.values())

            # Estimate how many rows we need so after filtering we have enough:
            n_prefetch = round(n_missing / sampled_ratio)
            if self.synthesis_batch_size:
                n_prefetch = min(n_prefetch, self.synthesis_batch_size)
            n_prefetch = min(n_prefetch, int(1e6))

            # Synthesis:
            df_synthesized = self.synthesizer.synthesize(num_rows=n_prefetch, conditions=conditions)

            # In order to filter our data frame we need keys that we will look up in counts:
            df_key = self.map_key_columns(df_synthesized, marginals_keys)

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

            if progress_callback is not None:
                progress_callback(round(len(result) * 100.0 / num_rows))

            del df_key, df_synthesized
            gc.collect()

        if progress_callback is not None:
            progress_callback(100)

        return pd.DataFrame.from_records(result, columns=self.all_columns).sample(frac=1).reset_index(drop=True)

    def alter_distributions(self,
                            df: pd.DataFrame,
                            num_rows: int,
                            explicit_marginals: Dict[str, Dict[str, float]] = None,
                            conditions: Union[dict, pd.DataFrame] = None,
                            progress_callback: Callable[[int], None] = None) -> pd.DataFrame:
        """Given a DataFrame, drop and/or generate new samples so that the output distributions are
         defined by explicit marginals.

        Args:
            df: Original DataFrame
            num_rows: The number of rows to generate.
            conditions: The condition values for the generated rows.
            progress_callback: Progress bar callback.
            explicit_marginals: A dict of desired marginal distributions per column.
                Distributions defined as density per category or bin. The result will be sampled
                from the synthesizer conditionally on these marginals.

        Returns:
            The generated data.

        """

        if progress_callback is not None:
            progress_callback(0)

        if explicit_marginals is None:
            return self.synthesizer.synthesize(num_rows, conditions=conditions, progress_callback=progress_callback)

        conditional_columns = list(explicit_marginals.keys())
        marginals_keys = {k: list(v.keys()) for k, v in explicit_marginals.items()}

        # For the sake of performance we will not really sample from "condition" distribution,
        # but will rather sample directly from synthesizer and filter records so they distribution is conditional
        self._validate_explicit_marginals(explicit_marginals)
        marginal_counts = self.get_joined_marginal_counts(explicit_marginals, num_rows)

        df_key = self.map_key_columns(df, marginals_keys)
        orig_key_groups = self._keys_to_tuple(df_key.groupby(conditional_columns).groups)

        marginal_counts_original_df = Counter({k: len(v) for k, v in orig_key_groups.items()})

        marginal_counts_to_synthesize: Dict[Tuple[str, ...], int] = Counter()
        marginal_counts_from_original: Dict[Tuple[str, ...], int] = Counter()

        for k in marginal_counts.keys():
            diff = marginal_counts[k] - marginal_counts_original_df[k]
            if diff > 0:
                marginal_counts_to_synthesize[k] = diff
                marginal_counts_from_original[k] = marginal_counts_original_df[k]
            elif diff < 0:
                marginal_counts_from_original[k] = marginal_counts[k]

        # Create an empty copy of the input DataFrame
        df_out = df.head(0)

        # Get rows from original dataset
        if len(marginal_counts_from_original) > 0:
            idx: List = []
            for k in marginal_counts_from_original.keys():
                idx.extend(np.random.choice(orig_key_groups[k], size=marginal_counts_from_original[k], replace=False))
            df_out = df_out.append(df[df.index.isin(idx)])

        # Synthesize missing rows
        if len(marginal_counts_to_synthesize) > 0:
            df_out = df_out.append(self.synthesize_from_joined_counts(
                marginal_counts_to_synthesize, marginals_keys, conditions=conditions,
                progress_callback=progress_callback
            ))

        return df_out.sample(frac=1).reset_index(drop=True)

    def map_key_columns(self, df: pd.DataFrame, marginals_keys: Dict[str, List[str]]) -> pd.DataFrame:
        """Get key dataframe. Transform the continuous columns into intervals, and convert all key
        columns into strings.

        """
        conditional_columns = list(marginals_keys.keys())
        df_key = df[conditional_columns]
        df_key = self._map_continuous_columns(df_key, marginals_keys)
        df_key = df_key.astype(str)
        return df_key

    def _map_continuous_columns(self, df: pd.DataFrame,
                                marginals_keys: Dict[str, List[str]]) -> pd.DataFrame:
        """Looks for continuous columns and map values into bins that are defined in `explicit_marginals`.

        Args:
            df: Input data frame.

        Returns:
            Result data frame.

        """
        df = df.copy()
        conditional_columns: List[str] = list(marginals_keys.keys())

        mapping = {}
        for col in self.continuous_columns:
            if col in conditional_columns:
                intervals = []
                for str_interval in marginals_keys[col]:
                    interval = FloatInterval.parse(str_interval)
                    intervals.append(interval)
                mapping[col] = intervals

        for col in conditional_columns:
            if col in self.continuous_columns:
                def map_value(value: float):
                    intervals = mapping[col]
                    for interval in intervals:
                        if interval.is_in(value):
                            return str(interval)
                df[col] = df[col].apply(map_value)

        return df

    def _validate_explicit_marginals(self, explicit_marginals: Dict[str, Dict[str, float]]):
        values = self.synthesizer.df_meta.values
        values_name = [value.name for value in values]

        for col, cond in explicit_marginals.items():
            if not np.isclose(sum(cond.values()), 1.0):
                raise ValueError("Marginal probabilities do not add up to 1 for '{}'".format(col))
            if col not in values_name:
                raise ValueError("Column '{}' not found in learned values for the given synthesizer.".format(col))

            v = values[values_name.index(col)]
            if isinstance(v, CategoricalValue):
                categories = [str(c) for c in v.categories if c is not np.nan]
                for category in cond.keys():
                    if not isinstance(category, str):
                        raise TypeError("Given bins must be strings. Bin {} is not a string".format(category))
                    elif category not in categories:
                        raise ValueError("Category '{}' for column '{}' not found in learned data. Available options "
                                         "are: '{}'".format(category, col, ', '.join(categories)))

    @staticmethod
    def get_joined_marginal_counts(explicit_marginals: Dict[str, Dict[str, float]],
                                   num_rows: int) -> Dict[Tuple, int]:
        # Let's compute cartesian product of all probs for each column
        # to get probs for the joined distribution:
        category_probs = []
        for column, distr in explicit_marginals.items():
            category_probs.append([(category, prob) for category, prob in distr.items()])
        category_combinations = product(*category_probs)
        rows = [
            tuple(zip(*comb))
            for comb in category_combinations
        ]
        joined_marginal_probs = {row[0]: np.product(row[1]) for row in rows}
        return Counter({c: int(round(p * num_rows)) for c, p in joined_marginal_probs.items()})

    @staticmethod
    def _keys_to_tuple(d: Dict[Union[str, Tuple[str, ...]], Any]) -> Dict[Tuple[str, ...], Any]:
        """For a given dict, ensure that keys are tuples of strings not strings"""

        new_dict: Dict[Tuple[str, ...], Any] = dict()
        for k in list(d.keys()):
            if isinstance(k, str):
                new_dict[(k,)] = d.pop(k)
            else:
                new_dict[k] = d[k]

        return new_dict

    def get_values(self) -> List[Value]:
        return self.synthesizer.get_values()

    def get_conditions(self) -> List[Value]:
        return self.synthesizer.get_conditions()

    def get_value_meta_pairs(self) -> List[Tuple[Value, ValueMeta]]:
        return self.synthesizer.get_value_meta_pairs()

    def get_condition_meta_pairs(self) -> List[Tuple[Value, ValueMeta]]:
        return self.synthesizer.get_condition_meta_pairs()

    def get_losses(self, data: Dict[str, tf.Tensor] = None) -> tf.Tensor:
        return self.synthesizer.get_losses()


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
