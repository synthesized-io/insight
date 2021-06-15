from abc import ABC, abstractmethod
from typing import Dict, Sequence, Type, Union

import pandas as pd

from synthesized.metadata import DataFrameMeta


class Unifier(ABC):
    """
    Base class for unifier part of the data oracle.
    """
    subclasses: Dict[str, Type['Unifier']] = {}

    def __init_subclass__(cls) -> None:
        cls.subclasses[cls.__name__] = cls

    @abstractmethod
    def update(self,
               dfs: Union[pd.DataFrame, Sequence[pd.DataFrame]] = None,
               df_metas: Union[DataFrameMeta, Sequence[DataFrameMeta]] = None,
               num_iterations: int = None) -> None:
        """
        Update adds a new dataframe to the Unifier object, this dataframe can then be used for Synthesis later on.

        Args:
            dfs: Single Dataframe or List of Dataframes that are to be incorporated into the Unifier.
                Either df or dfs should be provided
            df_metas: Single DataFrame meta or a list of DataFrame meta provided
            num_iterations: the number of iterations used to train the HighDimSynthesizer. Defaults to None, in
                which case the learning manager is used to determine when to stop training.
        """
        ...

    @abstractmethod
    def query(self, columns: Sequence[str], num_rows: int) -> pd.DataFrame:
        """
        Query the unifier for a dataframe of the made up from certain columns.

        Args:
            columns: Sequence of columns to make up the output dataframe
            num_rows: number of rows to generate in the output dataframe

        Returns:
            Synthesized dataframe with specified columns.
        """
        ...
