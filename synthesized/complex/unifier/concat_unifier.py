import pandas as pd
import functools

from typing import List, Union, Dict, Sequence, Optional

from .base import Unifier
from ...complex import HighDimSynthesizer
from ...metadata import DataFrameMeta
from ...metadata.factory import MetaExtractor


class ConcatUnifier(Unifier):
    """
    Concat unifier class concats given dataframes vertically using the common columns

    Attributes:
        self.store: A dictionary mapping dataframe name/idx to the dataframe
        self.meta: The dataframe meta of the unified datafrme
        self.synthesizer: Trained HighDimSynthesizer object for the unified dataframe
    """
    def __init__(self):
        self.store: Dict[str, pd.DataFrame] = {}
        self.meta: Optional[DataFrameMeta] = None
        self.synthesizer: Optional[HighDimSynthesizer] = None

    def _concat_dfs(self,
                    dfs: List[pd.DataFrame]):
        """Concatenates the given dataframes by stacking them vertically based on common columns"""
        intersection = functools.reduce(set.intersection, (set(df.columns) for df in dfs))
        if not intersection:
            raise ValueError("Cannot concatenate DataFrames that have no shared columns.")

        concatenated_df = pd.concat(dfs, axis=0, ignore_index=True)
        return concatenated_df

    def update(self,
               dfs: Union[pd.DataFrame, Sequence[pd.DataFrame]] = None,
               df_metas: Union[DataFrameMeta, Sequence[DataFrameMeta]] = None,
               num_iterations: int = None) -> None:
        """Modify the unifier by adding the given df/meta to the store
        and update the synthesizer object
        Args:
            dfs: Single Dataframe or List of Dataframes that are to be incorporated into the Unifier.
                Either df or dfs should be provided
            df_metas: Single DataFrame meta or a list of DataFrame meta provided
            num_iterations: the number of iterations used to train the HighDimSynthesizer. Defaults to None, in
                which case the learning manager is used to determine when to stop training.
        """
        if isinstance(dfs, pd.DataFrame):
            dfs = [dfs]

        if isinstance(df_metas, DataFrameMeta):
            df_metas = [df_metas]

        if dfs is not None and df_metas is not None:
            if len(dfs) == len(df_metas):
                cur_store_len = len(self.store.keys())
                for idx, df in enumerate(dfs):
                    self.store[f"df{cur_store_len+idx}"] = df
            else:
                raise ValueError("length of dfs and df_metas provided don't match")

        concatenated_df = self._concat_dfs(list(self.store.values()))
        self.meta = MetaExtractor.extract(concatenated_df)

        synth = HighDimSynthesizer(df_meta=self.meta)
        synth.learn(df_train=concatenated_df, num_iterations=num_iterations)
        self._synthesizer = synth

    def query(self,
              columns: Sequence[str],
              num_rows: int):
        """Query the unifier to synthesize given number of rows
        Args:
            num_rows: Number of rows to be synthesized
            columns: Columns to be returned in the synthesized dataframe
        """
        if self._synthesizer is None or self.meta is None:
            raise AttributeError('ConcatUnifier has not been updated.')
        elif self.meta.columns is not None and not all(c in self.meta.columns for c in columns):
            raise ValueError('ConcatUnifier has not seen the selected data.')

        synth = self._synthesizer.synthesize(num_rows)
        return synth[columns]
