from collections import Counter
from itertools import product
from typing import Any, Dict, Tuple, Union, Callable

import numpy as np
import pandas as pd

from ...synthesizer import Synthesizer


class ConditionalSampler(Synthesizer):
    def __init__(
            self, synthesizer: Synthesizer, *conditions: Tuple[str, Dict[Any, float]], min_fill_ratio: float = 0.001
    ):
        self.synthesizer = synthesizer
        self.columns = []
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
            all_columns = df_synthesized.columns
            n_added = 0
            columns_idx = [df_synthesized.columns.get_loc(c) for c in self.columns]
            for row in df_synthesized.to_numpy():
                key = tuple(row[columns_idx])
                if counts[key] > 0:
                    result.append(row)
                    n_added += 1
                    counts[key] -= 1
            if n_added == 0:
                fill_ratio = 1.0 / n_prefetch
            else:
                fill_ratio = float(n_added) / n_prefetch
        return pd.DataFrame.from_records(result, columns=all_columns)
