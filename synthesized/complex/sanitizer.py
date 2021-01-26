import logging
from collections import OrderedDict
from itertools import combinations
from typing import Any, Callable, Dict, List, Union

import numpy as np
import pandas as pd
import simplejson

from .highdim import HighDimConfig, HighDimSynthesizer
from ..common.synthesizer import Synthesizer
from ..common.values import CategoricalValue, ContinuousValue
from ..metadata_new import MetaExtractor

logger = logging.getLogger(__name__)


class Sanitizer(Synthesizer):
    """The default implementation. Drops duplicates. Floats are rounded."""

    def __init__(self,
                 synthesizer: Synthesizer,
                 df_original: pd.DataFrame,
                 distance_step: float = 1e-3) -> None:

        self.synthesizer = synthesizer
        self.df_original = df_original.copy()
        assert 0 < distance_step < 1
        self.distance_step = distance_step

        self.float_decimal = 5
        self.oversynthesis_ratio = 1.1
        self.max_synthesis_attempts = 3
        remove_outliers = 0.05

        self.learned_columns: List[str] = []
        self.categorical_values: List[str] = []
        self.distances: Dict[str, Union[float, None]] = OrderedDict()
        for v in self.synthesizer.get_values():
            if v.learned_output_size() > 0:
                self.learned_columns.append(v.name)

                if isinstance(v, ContinuousValue):
                    column = pd.to_numeric(df_original[v.name], errors='coerce').dropna()

                    # Remove outliers for distance computation
                    percentiles = [remove_outliers * 100. / 2, 100 - remove_outliers * 100. / 2]
                    start, end = np.percentile(column, percentiles)
                    if start == end:
                        start, end = min(column), max(column)
                    if start == end:
                        self.distances[v.name] = None

                    self.distances[v.name] = (end - start) * self.distance_step
                else:
                    if isinstance(v, CategoricalValue):
                        self.categorical_values.append(v.name)
                    self.distances[v.name] = None

        self.n_learned_columns = len(self.learned_columns)

    def _sanitize_all(self, df_synthesized: pd.DataFrame) -> pd.DataFrame:
        """Drop rows in df_synthesized that are present in df_original."""

        sorted_columns = list(self.distances.keys())
        sorted_distances = list(self.distances.values())

        def normalize_tuple(nt):
            res = []
            for i, field in enumerate(nt):
                if sorted_distances[i] is not None:
                    distance = sorted_distances[i]
                    field = round(field / distance) * distance
                res.append(field)
            return tuple(res)

        original_rows = {normalize_tuple(row) for row in self.df_original[sorted_columns].itertuples(index=False)}
        to_drop = []
        for i, row in enumerate(df_synthesized[sorted_columns].itertuples(index=False)):
            if normalize_tuple(row) in original_rows:
                to_drop.append(i)

        return df_synthesized.reset_index(drop=True).drop(to_drop)

    def _sanitize(self, df_synthesized: pd.DataFrame, n_cols: int = None,
                  skip_categorical: bool = True) -> pd.DataFrame:
        """Drop rows in df_synthesized that are present in df_original."""

        if n_cols is None:
            n_cols = self.n_learned_columns

        df_synthesized = df_synthesized.copy()
        if n_cols > self.n_learned_columns:
            raise ValueError("Given n_cols can't be larger than the number of learned columns ({}), "
                             "given {}".format(self.n_learned_columns, n_cols))

        if n_cols <= len(self.categorical_values):
            logger.warning("{}/{} columns are categorical or have few unique values. For given 'n_cols={}' there "
                           "may be strong overlap between original and synthetic data-sets"
                           .format(len(self.categorical_values), self.learned_columns, n_cols))

        n_dropped = 0
        initial_len = len(df_synthesized)

        df_original = self.df_original[self.learned_columns].drop_duplicates()

        for c in self.learned_columns:
            distance = self.distances[c]
            if distance is not None:
                df_original.loc[:, c] = df_original.loc[:, c].apply(lambda x: round(x / distance) * distance)
                df_synthesized.loc[:, c] = df_synthesized.loc[:, c].apply(lambda x: round(x / distance) * distance)

        for cols in combinations(self.learned_columns, n_cols):
            if skip_categorical and all([c in self.categorical_values for c in cols]):
                logger.debug("Skipping combination {} as it only contains categorical values".format(cols))
                continue
            original_rows = {row for row in df_original[list(cols)].itertuples(index=False)}

            to_drop = []

            for i, row in enumerate(df_synthesized[list(cols)].itertuples(index=False)):
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
            self, num_rows: int, produce_nans: bool = False,
            progress_callback: Callable[[int], None] = None, n_columns_intersect: int = None
    ) -> pd.DataFrame:

        if progress_callback is not None:
            progress_callback(0)

        df_synthesized = self.synthesizer.synthesize(num_rows=num_rows, produce_nans=produce_nans,
                                                     progress_callback=progress_callback)

        if progress_callback is not None:
            progress_callback(99)

        # the first drop of duplicates
        df_synthesized = self._sanitize(df_synthesized, n_cols=n_columns_intersect)

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
            df_additional = self.synthesizer.synthesize(num_rows=n_additional)
            df_additional = self._sanitize(df_additional, n_cols=n_columns_intersect)
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


def privacy_check(data: pd.DataFrame, num_iterations: int = None, synthesizer_class: str = 'HighDimSynthesizer',
                  n_cols: int = 3, distance_step: float = 1e-3, synthesizer_params: Dict[str, Any] = None,
                  return_json: bool = True) -> Union[str, List[Dict]]:

    data = data.copy()
    data.dropna(inplace=True)

    if synthesizer_params is None:
        synthesizer_params = dict()

    if synthesizer_class == 'HighDimSynthesizer':
        dp = MetaExtractor.extract(df=data)
        config = HighDimConfig(**synthesizer_params)
        synthesizer = HighDimSynthesizer(df_meta=dp, config=config)
        synthesizer.__enter__()
        synthesizer.learn(df_train=data, num_iterations=num_iterations)
    else:
        raise NotImplementedError

    sanitizer = Sanitizer(synthesizer, data, distance_step)
    learned_columns = sanitizer.learned_columns

    import time

    results = []
    for i in range(n_cols):
        d: Dict[str, Any] = dict()
        n_cols = len(learned_columns) - i
        synthesized = synthesizer.synthesize(num_rows=len(data))
        t_start = time.perf_counter()
        df_synthesized = sanitizer._sanitize(synthesized, n_cols=n_cols)
        took = time.perf_counter() - t_start
        logger.debug('took {:.2f}s'.format(took))

        d['n_cols'] = n_cols
        d['n_categorical'] = len(sanitizer.categorical_values)
        d['results'] = {
            'cleaned-rows-rate': (len(data) - len(df_synthesized)) / len(data),
            'cleaned-rows-num': len(data) - len(df_synthesized),
            'total-rows': len(data),
        }
        results.append(d)

    if return_json:
        return simplejson.dumps(results, indent='\t')
    else:
        return results
