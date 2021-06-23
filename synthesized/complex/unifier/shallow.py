from typing import Optional, Sequence, Union

import pandas as pd

from .base import Unifier
from ...metadata import DataFrameMeta
from ...model import DataFrameModel
from ...model.factory import ModelFactory


class ShallowOracle(Unifier):
    """Data oracle which uses statistical models and rules"""

    def __init__(self, df_model: Optional[DataFrameModel] = None):
        self.df_model = df_model

    @property
    def df_meta(self):
        return self.df_model.meta

    def update(
        self,
        dfs: Union[pd.DataFrame, Sequence[pd.DataFrame]],
        df_metas: Union[DataFrameMeta, Sequence[DataFrameMeta]],
        num_iterations: Optional[int] = None
    ) -> None:
        list_dfs: Sequence[pd.DataFrame] = [dfs] if isinstance(dfs, pd.DataFrame) else dfs
        list_df_metas = [df_metas] if isinstance(df_metas, DataFrameMeta) else df_metas

        if len(list_dfs) != len(list_df_metas):
            raise ValueError("Mismatching lengths of dfs and df_metas given.")

        for df, df_meta in zip(list_dfs, list_df_metas):
            self._update_df_model(df, df_meta)

    def _update_df_model(self, df: pd.DataFrame, df_meta: DataFrameMeta):

        if self.df_model is None:
            self.df_model = ModelFactory()(df_meta)
            self.df_model.fit(df)

        self.df_model.update_model(df)

        return None

    def query(self, columns: Sequence[str], num_rows: int) -> pd.DataFrame:
        """Query the oracle for a dataframe with the given columns"""
        if self.df_model is None:
            return pd.DataFrame(columns=[columns], index=pd.RangeIndex(num_rows))
        return self.df_model.sample(num_rows)[columns]
