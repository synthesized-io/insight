import gc
import logging
import re
from abc import ABC, abstractmethod
from collections import Counter
from datetime import datetime
from itertools import product
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from synthesized.complex import HighDimSynthesizer

import numpy as np
import pandas as pd
import tensorflow as tf

from .data_imputer import DataImputer
from ..common.synthesizer import Synthesizer
from ..common.values import CategoricalValue, ContinuousValue, DateValue, Value

logger = logging.getLogger(__name__)


class ConditionalSampler(Synthesizer):
    """Generate conditional data that corresponds to user-defined marginal distributions of specific columns.

    Allows for reshaping and of the original data distribution by defining custom marginal distributions for each
    column. This can be used for instance to upsample outlier data points and correct class imbalance.

    Args:
        synthesizer (Synthesizer): Sample from this trained Synthesizer instance.
        min_sampled_ratio (float, optional): Stop synthesis if the ratio of successfully sampled records is less than
            this value. Defaults to 0.001.
        synthesis_batch_size (int, optional): Generate data in batches of this size. Defaults to 65536. Larger values
            may speed up the generation process.

    Example:

        Initialise from a ``HighDimSynthesizer`` instance:

        >>> cond = ConditionalSampler(synthesizer)

        Define new marginal distributions for the ``age`` and ``SeriousDlqin2yrs`` columns:

        >>> marginals = {'SeriousDlqin2yrs': {'0': 0.3, '1': 0.7},
        >>>              'age': {'[0.0, 50.0)': 0.5, '[50.0, 100.0)': 0.5}}

        Synthesize 100 new rows that follow the defined marginal distributions:

        >>> cond.synthesize(num_rows=100, explicit_marginals=marginals))
    """

    def __init__(self,
                 synthesizer: 'HighDimSynthesizer',
                 min_sampled_ratio: float = 0.001,
                 synthesis_batch_size: Optional[int] = 65536):
        super().__init__(name='conditional')
        self.synthesizer = synthesizer
        self._global_step = synthesizer._global_step
        self.logdir = synthesizer.logdir
        self._loss_history = synthesizer._loss_history
        self._writer = synthesizer._writer

        self._min_sampled_ratio = min_sampled_ratio
        self._synthesis_batch_size = synthesis_batch_size

        self._date_fmt: Dict[str, Optional[str]] = dict()
        self._continuous_columns = []
        self._categorical_columns = []
        self._date_columns = []

        for v in self.synthesizer._df_value.values():
            if isinstance(v, DateValue):
                self._date_columns.append(v.name)
                self._date_fmt[v.name] = self.synthesizer.df_meta[v.name].date_format  # type: ignore
            elif isinstance(v, (CategoricalValue)):
                self._categorical_columns.append(v.name)
            elif isinstance(v, (ContinuousValue)):
                self._continuous_columns.append(v.name)

    def __repr__(self):
        return f"ConditionalSampler(synthesizer={self.synthesizer})"

    def learn(self, df_train: pd.DataFrame, num_iterations: Optional[int],
              callback: Optional[Callable[[object, int, dict], bool]] = None, callback_freq: int = 0) -> None:
        self.synthesizer.learn(
            num_iterations=num_iterations, df_train=df_train, callback=callback, callback_freq=callback_freq
        )

    def synthesize(self,
                   num_rows: int,
                   produce_nans: bool = False,
                   progress_callback: Callable[[int], None] = None,
                   explicit_marginals: Optional[Dict[str, Dict[str, float]]] = None,
                   date_fmt: Optional[str] = None) -> pd.DataFrame:
        """Generate the given number of new data rows according to user-defined marginal distributions.

        Custom distributions for each column can be specified using a dictionary structure, e.g
        ``{'binary_column_name': {'0': 0.5, '1': 0.5}}`` would produce a dataset where the `binary_column_name`
        feature has a uniform distribution across the two categories ``0`` and ``1``.

        For continuous features, the marginal distribution must be specified in terms of non-overlapping
        bins, e.g ``{'continuous_column_name'}: {'[0.0, 50.0)': 0.5, '[50.0, 100.0)': 0.5}}``. Each bin is defined by
        a string representation of the left and high right edges. See the examples for details.

        Args:
            num_rows (int): The number of rows to generate.
            produce_nans (bool): Whether to produce NaNs. Defaults to False.
            explicit_marginals (Dict[str, Dict[str, float]], optional): Desired marginal distributions per column,
                defined as probably density per category or bin. Defaults to None.
            date_fmt (str, optional): If conditons include dates, it's format. Defaults to None.
            progress_callback (Callable, optional): Progress bar callback.

        Returns:
            The generated data.

        Examples:

            Correct the class balance of a column with a severe class imbalance by defining a marginal distribution
            with a uniform distribution over the categories:

            >>> marginals = {'category': {'0': 0.5, '1': 0.5}}
            >>> cond = ConditonalSampler(synthesizer)

            Generate 100 rows of data in which there will be approximately a uniform distribution in the ``category``
            column:

            >>> cond.synthesize(num_rows=100, explicit_marginals=marginals)

        See Also:

            :meth:`ConditionalSampler.alter_distributions` :
                Adjust distributions of the original data with user-specified marginal distributions.

            :meth:`HighDimSynthesizer.synthesize_from_rules` :
                Generate data that corresponds to a specific set of conditions and constraints.
        """

        if progress_callback is not None:
            progress_callback(0)

        if explicit_marginals is None or len(explicit_marginals) == 0:
            return self.synthesizer.synthesize(num_rows, produce_nans=produce_nans,
                                               progress_callback=progress_callback)

        # For the sake of performance we will not really sample from "condition" distribution,
        # but will rather sample directly from synthesizer and filter records so they distribution is conditional
        self._validate_explicit_marginals(explicit_marginals)
        marginal_counts = self._get_joined_marginal_counts(explicit_marginals, num_rows)

        # Let's adjust counts so they sum up to `num_rows`:
        any_key = list(marginal_counts.keys())[0]
        marginal_counts[any_key] += num_rows - sum(marginal_counts.values())

        marginal_keys = {k: list(v.keys()) for k, v in explicit_marginals.items()}

        return self._synthesize_from_joined_counts(marginal_counts, marginal_keys, date_fmt=date_fmt,
                                                   progress_callback=progress_callback)

    def _synthesize_from_joined_counts(
            self,
            marginal_counts: Dict[Tuple[str, ...], int],
            marginal_keys: Dict[str, List[str]],
            date_fmt: Optional[str] = None,
            produce_nans: bool = False,
            progress_callback: Callable[[int], None] = None,
            max_trials: Optional[int] = 20,
    ) -> pd.DataFrame:
        """Given joint counts, synthesize dataset."""

        # TODO: Remove not learned columns from marginal_counts & marginal_keys

        columns = self.synthesizer.df_meta.columns
        marginal_counts = marginal_counts.copy()
        self._validate_marginal_counts_and_keys(marginal_counts, marginal_keys)

        max_n_prefetch = int(1e6)
        num_rows = sum(marginal_counts.values())
        if max_trials is not None and num_rows > max_n_prefetch * max_trials:
            logger.warning(f"Given total number of rows to generate is limited to {max_n_prefetch * max_trials}, "
                           f"will generate less samples than asked for.")

        contains_colons = any([v == ":" for k in marginal_counts.keys() for v in k])
        num_rows = sum(marginal_counts.values())

        # The result is a list of result arrays
        result = []

        n_trials = 0
        n_trials_non_added = 0
        n_missing = prev_n_missing = sum(marginal_counts.values())
        sampled_ratio = 1.01
        while sum(marginal_counts.values()) > 0 and sampled_ratio >= self._min_sampled_ratio:

            # Estimate how many rows we need so after filtering we have enough:
            n_prefetch = round(n_missing / sampled_ratio)
            if self._synthesis_batch_size:
                n_prefetch = min(n_prefetch, self._synthesis_batch_size)
            n_prefetch = min(n_prefetch, max_n_prefetch)

            # Synthesis:
            df_synthesized = self.synthesizer.synthesize(num_rows=n_prefetch,
                                                         produce_nans=produce_nans)

            # In order to filter our data frame we need keys that we will look up in counts:
            df_key = self._map_key_columns(df_synthesized, marginal_keys, date_fmt=date_fmt)

            n_added = 0
            for key_row, row in zip(df_key.to_numpy(), df_synthesized.to_numpy()):
                key: Tuple[str, ...] = tuple(key_row)

                # If counter for the instance is positive let's emit the current row:
                if contains_colons:
                    for marginal_key, counts in marginal_counts.items():
                        if all([marginal_value_i == ":" or marginal_value_i == key[i] for i, marginal_value_i in
                                enumerate(marginal_key)]) and counts > 0:
                            result.append(row)
                            n_added += 1
                            marginal_counts[marginal_key] -= 1

                else:  # Better to keep them separated as this is faster in the case where there are no colons
                    if marginal_counts[key] > 0:
                        result.append(row)
                        n_added += 1
                        marginal_counts[key] -= 1

            if n_added == 0:
                # In case if we couldn't sample anything this time:
                sampled_ratio = 1.0 / n_prefetch
            else:
                sampled_ratio = float(n_added) / n_prefetch

            n_trials += 1
            if max_trials is not None and n_trials >= max_trials:
                logger.warning(f"Synthesis stopped after {n_trials} trials being able to generate {n_added} samples.")
                break

            # If after 3 loops we were not able to generate anything,
            n_trials_non_added = 0 if n_missing != prev_n_missing else n_trials_non_added + 1
            if n_trials_non_added > 3:
                logger.warning("Synthesis stopped after 3 trials without being able to generate any samples.")
                break

            prev_n_missing = n_missing
            n_missing = sum(marginal_counts.values())
            logger.debug(f"Loop finished, n_added={n_added}, n_missing={n_missing}")

            # Free memory
            del df_key, df_synthesized
            gc.collect()

            if progress_callback is not None:
                progress_callback(round(len(result) * 100.0 / num_rows))

        if progress_callback is not None:
            progress_callback(100)

        df_synth = pd.DataFrame.from_records(result, columns=columns).sample(frac=1).reset_index(drop=True)

        # Set same dtypes as input
        self.synthesizer._df_transformer.set_dtypes(df_synth)
        return df_synth

    def alter_distributions(
            self,
            df: pd.DataFrame,
            num_rows: int,
            produce_nans: bool = False,
            explicit_marginals: Dict[str, Dict[str, float]] = None,
            date_fmt: Optional[str] = None,
            progress_callback: Callable[[int], None] = None,
    ) -> pd.DataFrame:
        """Given a DataFrame, drop and/or generate new samples so that the column distributions are
        defined by user-specified marginals distributions. Unlike the :meth:`ConditionalSampler.synthesize`
        method, this will keep some of the original data, and therefore the output will not be purely synthetic data.

        Args:
            df (pd.DataFrame): DataFrame of original data to modify.
            num_rows (int): The number of rows to generate.
            date_fmt (str, optional): If the conditons include dates, it's format. Defaults to None.
            produce_nans (bool): Whether to produce NaNs. Defaults to False.
            progress_callback (Callable, optional): Progress bar callback. Defaults to None.
            explicit_marginals (List[Dict[str, Dict[str, float]]]): Desired marginal distributions per column,
                defined as probably density per category or bin.

        Returns:
            The generated data.

        See Also:
            :meth:`ConditionalSampler.synthesize` : Generate synthetic data from user-specified marginal distributions.
        """

        if progress_callback is not None:
            progress_callback(0)

        if explicit_marginals is None:
            return self.synthesizer.synthesize(num_rows, produce_nans=produce_nans,
                                               progress_callback=progress_callback)

        conditional_columns = list(explicit_marginals.keys())
        marginal_keys = {k: list(v.keys()) for k, v in explicit_marginals.items()}

        # For the sake of performance we will not really sample from "condition" distribution,
        # but will rather sample directly from synthesizer and filter records so they distribution is conditional
        self._validate_explicit_marginals(explicit_marginals)
        marginal_counts = self._get_joined_marginal_counts(explicit_marginals, num_rows)

        df_key = self._map_key_columns(df, marginal_keys, date_fmt=date_fmt)
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
                if marginal_counts_from_original[k] > 0:
                    idx.extend(np.random.choice(orig_key_groups[k], size=marginal_counts_from_original[k],
                                                replace=False))

            df_sample = df[df.index.isin(idx)]

            # Impute nans if we want the output to not have nans
            if not produce_nans and any(df_sample.isna()):
                data_imputer = DataImputer(self.synthesizer)
                data_imputer.impute_nans(df_sample, inplace=True)

            df_out = df_out.append(df_sample)

        # Synthesize missing rows
        if len(marginal_counts_to_synthesize) > 0:
            df_out = df_out.append(self._synthesize_from_joined_counts(
                marginal_counts_to_synthesize, marginal_keys, produce_nans=produce_nans,
                progress_callback=progress_callback
            ))

        return df_out.sample(frac=1).reset_index(drop=True)

    def _map_key_columns(self, df: pd.DataFrame, marginal_keys: Dict[str, List[str]],
                         date_fmt: Optional[str] = None) -> pd.DataFrame:
        """Get key dataframe. Transform the continuous columns into intervals, and convert all key
        columns into strings.

        """
        conditional_columns = list(marginal_keys.keys())
        df_key = df[conditional_columns]
        df_key = self._map_continuous_columns(df_key, marginal_keys, date_fmt=date_fmt)
        df_key = df_key.astype(str)
        return df_key

    def _map_continuous_columns(self, df: pd.DataFrame, marginal_keys: Dict[str, List[str]],
                                date_fmt: Optional[str] = None) -> pd.DataFrame:
        """Looks for continuous columns and map values into bins that are defined in `explicit_marginals`.

        Args:
            df: Input data frame.

        Returns:
            Result data frame.

        """
        df = df.copy()
        conditional_columns: List[str] = list(marginal_keys.keys())

        # Find float -> str mappings
        mapping = {}
        for col in self._continuous_columns:
            if col in conditional_columns:
                intervals = []
                for str_interval in marginal_keys[col]:
                    interval = FloatInterval.parse(str_interval)
                    intervals.append(interval)
                mapping[col] = intervals

        # Find date -> str mappings
        for col in self._date_columns:
            if col in conditional_columns:
                date_fmt = date_fmt if date_fmt is not None else self._date_fmt[col]
                if date_fmt is None:
                    raise ValueError(f"Could not infer date format for column '{col}', please provide it")
                intervals = []
                for str_interval in marginal_keys[col]:
                    interval = DateInterval.parse(str_interval, date_fmt=date_fmt)
                    intervals.append(interval)
                mapping[col] = intervals

        # Apply mappings
        for col in conditional_columns:
            if col in np.concatenate((self._continuous_columns, self._date_columns)):
                def map_value(value: float) -> str:
                    intervals = mapping[col]
                    for interval in intervals:
                        if interval.is_in(value):
                            return str(interval)
                    return ''

                if col in self._date_columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                df[col] = df[col].apply(map_value)

        return df

    def _validate_explicit_marginals(self, explicit_marginals: Dict[str, Dict[str, float]]) -> None:
        values_name = [value for value in self.synthesizer._df_value.keys()]

        for col, cond in explicit_marginals.items():
            if not np.isclose(sum(cond.values()), 1.0):
                raise ValueError("Marginal probabilities do not add up to 1 for '{}'".format(col))
            if col not in values_name:
                raise ValueError("Column '{}' not found in learned values for the given synthesizer.".format(col))

            if col in self._categorical_columns:
                for category in cond.keys():
                    if not isinstance(category, str):
                        raise TypeError("Given bins must be strings. Bin {} is not a string".format(category))

    @staticmethod
    def _validate_marginal_counts_and_keys(marginal_counts: Dict[Tuple[str, ...], int],
                                           marginal_keys: Dict[str, List[str]]) -> None:
        n_keys = len(marginal_keys)
        if not all([n_keys == len(k) for k in marginal_counts.keys()]):
            raise ValueError("The length of the keys of all 'marginal_counts' and 'marginal_keys' must be equal")

        ConditionalSampler._validate_marginal_counts(marginal_counts)
        ConditionalSampler._validate_marginal_keys(marginal_keys)

    @staticmethod
    def _validate_marginal_counts(marginal_counts: Dict[Tuple[str, ...], int]) -> None:
        """Validate datatypes of marginal_counts"""

        if not isinstance(marginal_counts, Counter):
            raise TypeError(f"Given 'marginal_counts' must be type 'Counter', given '{type(marginal_counts)}'")

        for marginals, counts in marginal_counts.items():
            # Check marginals
            if not isinstance(marginals, tuple):
                raise TypeError(f"Given key '{marginals}' of 'marginal_counts' must be type 'tuple', "
                                f"given '{type(marginals)}'")
            for marginal in marginals:
                if not isinstance(marginal, str):
                    raise TypeError(f"Given marginal '{marginal}' must be type 'str', "
                                    f"given '{type(marginal)}'")

            # Check counts
            if not isinstance(counts, int):
                raise TypeError(f"Given counts '{counts}' for marginals '{marginals}' must be type 'int', "
                                f"given '{type(counts)}'")

    @staticmethod
    def _validate_marginal_keys(marginal_keys: Dict[str, List[str]]) -> None:
        """Validate datatypes of marginal_counts"""

        if not isinstance(marginal_keys, dict):
            raise TypeError(f"Given 'marginal_keys' must be type 'dict', given '{type(marginal_keys)}'")

        for marginal, keys in marginal_keys.items():
            if not isinstance(marginal, str):
                raise TypeError(f"Given key '{marginal}' of 'marginal_keys' must be type 'str', "
                                f"given '{type(marginal)}'")

            if not isinstance(keys, list):
                raise TypeError(f"Given value '{keys}' of 'marginal_keys' must be type 'list', "
                                f"given '{type(keys)}'")

            for key in keys:
                if not isinstance(key, str):
                    raise TypeError(f"Given element '{key}' of 'marginal_keys[{marginal}]' must be type 'str', "
                                    f"given '{type(key)}'")

    @staticmethod
    def _get_joined_marginal_counts(explicit_marginals: Dict[str, Dict[str, float]],
                                    num_rows: int) -> Dict[Tuple[str, ...], int]:
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

    def _get_values(self) -> List[Value]:
        return self.synthesizer._get_values()

    def _get_losses(self, data: Dict[str, tf.Tensor] = None) -> tf.Tensor:
        return self.synthesizer._get_losses()


class Endpoint(ABC):
    def __init__(self, value: Union[float, datetime], value_str: Optional[str] = None):
        self.value = value
        self.value_str = value_str if value_str is not None else str(value)

    @abstractmethod
    def to_str(self, is_left: bool) -> str:
        pass

    @abstractmethod
    def as_left_in(self, value: Union[float, datetime]) -> bool:
        pass

    @abstractmethod
    def as_right_in(self, value: Union[float, datetime]) -> bool:
        pass


class Inclusive(Endpoint):
    def __init__(self, value: Union[float, datetime], value_str: Optional[str] = None):
        super().__init__(value, value_str=value_str)

    def as_left_in(self, value: Union[float, datetime]) -> bool:
        return value >= self.value  # type: ignore

    def as_right_in(self, value: Union[float, datetime]) -> bool:
        return value <= self.value  # type: ignore

    def to_str(self, is_left: bool) -> str:
        if is_left:
            return '[{}'.format(self.value_str)
        else:
            return '{}]'.format(self.value_str)

    def __eq__(self, other):
        return self.value == other.value

    def __repr__(self):
        return "Inclusive({})".format(self.value)


class Exclusive(Endpoint):
    def __init__(self, value: Union[float, datetime], value_str: Optional[str] = None):
        super().__init__(value, value_str=value_str)

    def as_left_in(self, value: Union[float, datetime]) -> bool:
        return value > self.value  # type: ignore

    def as_right_in(self, value: Union[float, datetime]) -> bool:
        return value < self.value  # type: ignore

    def to_str(self, is_left: bool) -> str:
        if is_left:
            return '({}'.format(self.value_str)
        else:
            return '{})'.format(self.value_str)

    def __eq__(self, other):
        return self.value == other.value

    def __repr__(self):
        return "Exclusive({})".format(self.value)


class FloatInterval:
    """Models an interval of float values."""

    RE = re.compile(r'([\[\(])(\S+\.\S+),\s(\S+\.\S+)([\]\)])')

    def __init__(self, left: Endpoint, right: Endpoint):
        assert left.value < right.value  # type: ignore
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

    @staticmethod
    def create_left_endpoint(left: Union[float, datetime], left_bracket: str,
                             left_s: Optional[str] = None) -> Endpoint:
        if left_bracket == '[':
            left_endpoint: Endpoint = Inclusive(left, left_s)
        elif left_bracket == '(':
            left_endpoint = Exclusive(left, left_s)
        else:
            assert False

        return left_endpoint

    @staticmethod
    def create_right_endpoint(right: Union[float, datetime], right_bracket: str,
                              right_s: Optional[str] = None) -> Endpoint:

        if right_bracket == ']':
            right_endpoint: Endpoint = Inclusive(right, right_s)
        elif right_bracket == ')':
            right_endpoint = Exclusive(right, right_s)
        else:
            assert False

        return right_endpoint

    @classmethod
    def parse(cls, s: str) -> 'FloatInterval':
        m = FloatInterval.RE.match(s)
        if m is None:
            raise ValueError(f"Can't match given string '{s}'")

        left_bracket, left_s, right_s, right_bracket = m.groups()
        left, right = float(left_s), float(right_s)

        left_endpoint = FloatInterval.create_left_endpoint(left, left_bracket, left_s)
        right_endpoint = FloatInterval.create_right_endpoint(right, right_bracket, right_s)
        return cls(left_endpoint, right_endpoint)


class DateInterval(FloatInterval):
    """Models an interval of date values."""

    def __init__(self, left: Endpoint, right: Endpoint, date_fmt: str = "%Y-%m-%d"):
        super().__init__(left=left, right=right)
        self.date_fmt = date_fmt

    def is_in(self, value: Union[str, datetime]) -> bool:  # type: ignore
        value_date: datetime = datetime.strptime(value, self.date_fmt) if isinstance(value, str) else value
        return self.left.as_left_in(value_date) and self.right.as_right_in(value_date)

    def __repr__(self):
        return 'DateInterval({}, {})'.format(self.left, self.right)

    @staticmethod
    def get_interval_fmt(date_fmt: str = "%Y-%m-%d"):
        date_fmt_re = re.sub(r"%\w", r"[0-9]+", date_fmt)
        return re.compile(r"([\[\(])({0}),\s({0})([\]\)])".format(date_fmt_re))

    @classmethod
    def parse(cls, s: str, date_fmt: str = "%Y-%m-%d") -> 'DateInterval':

        date_re = DateInterval.get_interval_fmt(date_fmt)
        m = date_re.match(s)
        if m is None:
            raise ValueError(f"Can't match given string '{s}'")

        left_bracket, left_s, right_s, right_bracket = m.groups()
        left, right = datetime.strptime(left_s, date_fmt), datetime.strptime(right_s, date_fmt)
        left_endpoint = FloatInterval.create_left_endpoint(left, left_bracket, left_s)
        right_endpoint = FloatInterval.create_right_endpoint(right, right_bracket, right_s)

        return cls(left_endpoint, right_endpoint)
