from itertools import combinations
import logging
from typing import Union, Callable

import pandas as pd

from ..synthesizer import Synthesizer

logger = logging.getLogger(__name__)


class Sanitizer(Synthesizer):
    """The default implementation. Drops duplicates. Floats are rounded."""

    def __init__(self,
                 synthesizer: Synthesizer,
                 df_original: pd.DataFrame) -> None:

        self.synthesizer = synthesizer
        self.df_original = df_original.copy()

        self.float_decimal = 5
        self.oversynthesis_ratio = 1.1
        self.max_synthesis_attempts = 3

        self.learned_columns = []
        for v in self.synthesizer.get_values():
            if v.learned_output_size() > 0:
                self.learned_columns.append(v.name)

    def _sanitize_all(self, df_synthesized: pd.DataFrame) -> pd.DataFrame:
        """Drop rows in df_synthesized that are present in df_original."""

        def normalize_tuple(nt):
            res = []
            for field in nt:
                if isinstance(field, float):
                    field = round(field, self.float_decimal)
                res.append(field)
            return tuple(res)

        original_rows = {normalize_tuple(row) for row in self.df_original.itertuples(index=False)}
        to_drop = []
        for i, row in enumerate(df_synthesized.itertuples(index=False)):
            if normalize_tuple(row) in original_rows:
                to_drop.append(i)

        return df_synthesized.reset_index(drop=True).drop(to_drop)

    def _sanitize(self, df_synthesized: pd.DataFrame, n_cols: int = None) -> pd.DataFrame:
        """Drop rows in df_synthesized that are present in df_original."""

        df_synthesized = df_synthesized.copy()
        if n_cols and n_cols > len(df_synthesized.columns):
            raise ValueError("Given n_cols can't be larger than the number of columnsin the dataframe, given {}".format(n_cols))

        n_dropped = 0
        initial_len = len(df_synthesized)

        if n_cols is None or n_cols >= len(self.learned_columns):
            df_synthesized = self._sanitize_all(df_synthesized)
            n_dropped = initial_len - len(df_synthesized)
            logger.debug('Total num. of dropped samples: {} / {} ({:.2f}%)'.format(n_dropped, initial_len,
                                                                                   n_dropped / initial_len * 100))
            return df_synthesized

        df_original = self.df_original[self.learned_columns].drop_duplicates()

        for c in self.learned_columns:
            if df_original[c].dtype.kind == 'f':
                df_original.loc[:, c] = df_original.loc[:, c].apply(lambda x: round(x, self.float_decimal))
                df_synthesized.loc[:, c] = df_synthesized.loc[:, c].apply(lambda x: round(x, self.float_decimal))

        for cols in combinations(self.learned_columns, n_cols):
            cols = list(cols)
            original_rows = {row for row in df_original[cols].itertuples(index=False)}

            to_drop = []

            for i, row in enumerate(df_synthesized[cols].itertuples(index=False)):
                if row in original_rows:
                    to_drop.append(i)

            if len(to_drop) > 0:
                n_dropped += len(to_drop)
                df_synthesized.reset_index(drop=True, inplace=True)
                df_synthesized.drop(to_drop, inplace=True)

        logger.debug('Total num. of dropped samples: {} / {} ({:.2f}%)'.format(n_dropped, initial_len,
                                                                               n_dropped / initial_len * 100))
        return df_synthesized

    def synthesize(
            self, num_rows: int, conditions: Union[dict, pd.DataFrame] = None,
            progress_callback: Callable[[int], None] = None
    ) -> pd.DataFrame:

        if progress_callback is not None:
            progress_callback(0)

        df_synthesized = self.synthesizer.synthesize(num_rows=num_rows, conditions=conditions,
                                                     progress_callback=progress_callback)

        if progress_callback is not None:
            progress_callback(99)

        # the first drop of duplicates
        df_synthesized = self._sanitize(df_synthesized)

        # we will use fill_ratio to predict how many more records we need
        fill_ratio = len(df_synthesized) / float(num_rows)
        if fill_ratio == 0:
            raise ValueError("All synthesized samples are in the original dataset.")

        attempt = 0

        # we will repeat synthesis and dropping unless we have enough records
        while len(df_synthesized) < num_rows:
            attempt += 1

            # we computer how many rows are missing and use fill_ratio to predict how many we will synthesize
            # also, we slightly increase this number by OVERSYNTHESIS_RATIO to get the result quicker
            n_additional = round((num_rows - len(df_synthesized)) / fill_ratio * self.oversynthesis_ratio)

            # synthesis + dropping
            df_additional = self.synthesizer.synthesize(num_rows=n_additional, conditions=conditions)
            df_additional = self._sanitize(df_additional)
            df_synthesized = df_synthesized.append(df_additional, ignore_index=True)

            # we give up after some number of attempts
            if attempt >= self.max_synthesis_attempts:
                break

        if progress_callback is not None:
            progress_callback(100)

        if len(df_synthesized) > num_rows:
            return df_synthesized.sample(num_rows)
        elif len(df_synthesized) == num_rows:
            return df_synthesized
        else:
            logger.warning("After {} attempts, the number of synthesized rows is fewer than 'num_rows' as there is an "
                           "overlap between synthetic and original data.".format(attempt))
            return df_synthesized
