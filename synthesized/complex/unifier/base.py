from abc import ABC, abstractmethod
from typing import Sequence

import pandas as pd


class Unifier(ABC):
    """
    Base class for unifier part of the data oracle.
    """

    def __init__(self):
        ...

    @abstractmethod
    def update(self, df: pd.DataFrame) -> None:
        """
        Update adds a new dataframe to the Unifier object, this dataframe can then be used for Synthesis later on.

        Args:
            df: Dataframe that is incorporated into the Unifier for later querying
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
