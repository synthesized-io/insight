from typing import Union, Callable

import pandas as pd

from ...synthesizer import Synthesizer


class Sanitizer(Synthesizer):
    """The default implementation. Drops duplicates. Floats are rounded."""

    FLOAT_DECIMAL = 5
    OVERSYNTHESIS_RATIO = 1.1
    MAX_SYNTHESIS_ATTEMPTS = 3

    def __init__(self,
                 synthesizer: Synthesizer,
                 df_original: pd.DataFrame) -> None:

        self.synthesizer = synthesizer
        self.df_original = df_original

    def sanitize(self, df_synthesized: pd.DataFrame) -> pd.DataFrame:
        """Drop rows in df_synthesized that are present in df_original."""

        def normalize_tuple(nt):
            res = []
            for field in nt:
                if isinstance(field, float):
                    field = round(field, Sanitizer.FLOAT_DECIMAL)
                res.append(field)
            return tuple(res)

        original_rows = {normalize_tuple(row) for row in self.df_original.itertuples(index=False)}
        to_drop = []
        for i, row in enumerate(df_synthesized.itertuples(index=False)):
            if normalize_tuple(row) in original_rows:
                to_drop.append(i)
        return df_synthesized.drop(to_drop)

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
        df_synthesized = self.sanitize(df_synthesized)

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
            n_additional = round((num_rows - len(df_synthesized)) / fill_ratio * Sanitizer.OVERSYNTHESIS_RATIO)

            # synthesis + dropping
            df_additional = self.synthesizer.synthesize(num_rows=n_additional, conditions=conditions)
            df_additional = self.sanitize(df_additional)
            df_synthesized = df_synthesized.append(df_additional, ignore_index=True)

            # we give up after some number of attempts
            if attempt >= Sanitizer.MAX_SYNTHESIS_ATTEMPTS:
                break

        if progress_callback is not None:
            progress_callback(100)

        if len(df_synthesized) >= num_rows:
            return df_synthesized.sample(num_rows)
        else:
            return df_synthesized
