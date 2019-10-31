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
    def __init__(
            self, synthesizer: Synthesizer, *conditions: Tuple[str, Dict[Any, float]], min_fill_ratio: float = 0.001
    ):
        self.synthesizer = synthesizer
        self.conditions: Dict[str, Dict[Any, float]] = {}
        for col, cond in conditions:
            self.conditions[col] = cond
        self.columns: List[str] = []
        category_probs = []
        for column, distr in conditions:
            self.columns.append(column)
            category_probs.append([(category, prob) for category, prob in distr.items()])
        category_combinations = product(*category_probs)
        rows = [
            tuple(zip(*comb))
            for comb in category_combinations
        ]
        self.joined_probs = {row[0]: np.product(row[1]) for row in rows}
        self.min_fill_ratio = min_fill_ratio

    def learn(self, num_iterations: int, df_train: pd.DataFrame,
              callback: Callable[[object, int, dict], bool] = Synthesizer.logging, callback_freq: int = 0) -> None:
        self.synthesizer.learn(
            num_iterations=num_iterations, df_train=df_train, callback=callback, callback_freq=callback_freq
        )

    def synthesize(self, num_rows: int, conditions: Union[dict, pd.DataFrame] = None):
        counts = Counter({c: int(round(p * num_rows)) for c, p in self.joined_probs.items()})
        result = []
        fill_ratio = 1.0
        all_columns = None
        while sum(counts.values()) > 0 and fill_ratio > self.min_fill_ratio:
            n_missing = sum(counts.values())
            n_prefetch = round(n_missing / fill_ratio * 1.1)
            n_prefetch = min(n_prefetch, int(1e6))
            df_synthesized = self.synthesizer.synthesize(num_rows=n_prefetch, conditions=conditions)
            df_key = df_synthesized[self.columns]
            df_key = self._map_continuous_columns(df_key)
            all_columns = df_synthesized.columns
            n_added = 0
            for key_row, row in zip(df_key.to_numpy(), df_synthesized.to_numpy()):
                key = tuple(key_row)
                if counts[key] > 0:
                    result.append(row)
                    n_added += 1
                    counts[key] -= 1
            if n_added == 0:
                fill_ratio = 1.0 / n_prefetch
            else:
                fill_ratio = float(n_added) / n_prefetch
        return pd.DataFrame.from_records(result, columns=all_columns)

    def _map_continuous_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        mapping = {}
        continuos_columns = {v.name for v in self.synthesizer.values if isinstance(v, ContinuousValue)}  # type: ignore
        for col in continuos_columns:
            if col in self.conditions:
                intervals = []
                for str_interval in self.conditions[col].keys():
                    interval = FloatInterval.parse(str_interval)
                    intervals.append(interval)
                mapping[col] = intervals

        for col in self.columns:
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

    def to_str(self, is_left: bool):
        pass

    def as_left_in(self, value: float):
        pass

    def as_right_in(self, value: float):
        pass


class Inclusive(FloatEndpoint):
    def __init__(self, value: float):
        super().__init__(value)

    def as_left_in(self, value: float):
        return value >= self.value

    def as_right_in(self, value: float):
        return value <= self.value

    def to_str(self, is_left: bool):
        if is_left:
            return '[{}'.format(self.value)
        else:
            return '{}]'.format(self.value)


class Exclusive(FloatEndpoint):
    def __init__(self, value: float):
        super().__init__(value)

    def as_left_in(self, value: float):
        return value > self.value

    def as_right_in(self, value: float):
        return value < self.value

    def to_str(self, is_left: bool):
        if is_left:
            return '({}'.format(self.value)
        else:
            return '{})'.format(self.value)


class FloatInterval:
    RE = re.compile(r'([\[\(])(\S+\.\S+),\s(\S+\.\S+)([\]\)])')

    def __init__(self, left: FloatEndpoint, right: FloatEndpoint):
        self.left = left
        self.right = right

    def is_in(self, value: float):
        return self.left.as_left_in(value) and self.right.as_right_in(value)

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
