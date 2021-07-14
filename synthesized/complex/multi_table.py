from typing import Callable, Dict, Optional, Tuple

import pandas as pd

from . import HighDimSynthesizer
from ..common.synthesizer import Synthesizer
from ..metadata import DataFrameMeta
from ..metadata.factory import MetaExtractor


class TwoTableSynthesizer(Synthesizer):
    """Synthesizer that can learn to generate data from two tables.

    The tables can have a one-to-one or one-to-many relationship and are linked with a single primary and foreign key.
    This ensures that the generated data:

    1. Keeps the referential integrity between the two tables, i.e the primary and foreign keys can be joined.
    2. Accurately generates the distribution of the foreign key counts in the case of a one-to-many relationship.

    It is assumed that each row in the first table is a unique entity, such as a customer, with no duplicates. The
    second table must relate to the first through a foreign key that matches the primary key, e.g the history of
    purchases for each customer.

    Args:
        df_meta ([Tuple[DataFrameMeta, DataFrameMeta]]): A tuple of the extracted DataFrameMeta for the two tables.
            Both tables must have a unique primary key, and the second table is assumed to have a many-to-one
            relationship with the first table with a corresponding unique foreign key.
        keys: Tuple[str]: The column names that identify the primary keys of each table. The primary key of the first
            table must exist in the second table as as foreign key.
        relation (Dict[str, str], Optional): A dictionary that maps the primary key column name of the first table
            to the foreign key column name in the second table if they are not identical. Defaults to None.

    Example:

        Load two tables that have a primary and foreign key relation.

        >>> df_customer = pd.read_csv('customer_table.csv')
        >>> df_transactions = pd.read_csv('transaction_table.csv')

        Extract the DataFrameMeta for each table:

        >>> df_metas = (MetaExtractor.extract(df_cust), MetaExtractor.extract(df_tran))

        Initialise the ``TwoTableSynthesizer``. The column names of the primary keys of each table are specified using
        ``keys`` parameter. The foreign key in df_transaction is 'customer_id', and this has the same column name as
        the primary key in df_customer":

        >>> synthesizer = TwoTableSynthesizer(df_metas=df_metas, keys=('customer_id', 'transaction_id'))

        Train the Synthesizer:

        >>> synthesizer.learn(df_train=dfs)

        Generate 1000 rows of new data:

        >>> df_customer_synthetic, df_transaction_synthetic = synthesizer.synthesize(num_rows=1000)
    """
    def __init__(
            self, df_metas: Tuple[DataFrameMeta, DataFrameMeta], keys: Tuple[str, str], relation: Dict[str, str] = None
    ) -> None:

        self.df_metas = df_metas
        self.keys = keys
        self.relation = {keys[0]: keys[0]} if relation is None else relation

        for name, key, meta in zip(*self.relation.items(), self.keys, df_metas):
            if key not in meta:
                raise ValueError(f"{key} is not a valid column name.")

        self._synthesizers = [HighDimSynthesizer(df_meta) for df_meta in df_metas]

    def __repr__(self):
        return f"TwoTableSynthesizer(df_metas={self.df_metas}, keys={self.keys}, relation={self.relation})"

    def learn(
            self, df_train: Tuple[pd.DataFrame, pd.DataFrame], num_iterations: Optional[int] = None,
            callback: Callable[[Synthesizer, int, dict], bool] = None,
            callback_freq: int = 0
    ) -> None:
        """Train the TwoTableSynthesizer.

        Args:
            df_train (Tuple[pd.DataFrame]): The training data for each table.
            callback: A callback function, e.g. for logging purposes. Takes the synthesizer
                instance, the iteration number, and a dictionary of values (usually the losses) as
                arguments. Aborts training if the return value is True.
            callback_freq: Callback frequency.
        """

        df_1, df_2 = self._build_relation(df_train)

        for df_train, synth in zip((df_1, df_2), self._synthesizers):
            df_meta = MetaExtractor.extract(df_train)
            synth._init_engine(df_meta, type_overrides=synth.type_overrides)
            synth.learn(df_train, num_iterations=num_iterations, callback=callback, callback_freq=callback_freq)

    def synthesize(
            self, num_rows: int, produce_nans: bool = False, progress_callback: Callable[[int], None] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate the given number of new data rows for table 1, and the associated rows of table 2

        Args:
            num_rows: The number of rows to generate.
            produce_nans: Whether to produce NaNs.
            progress_callback: Progress bar callback.

        Returns:
            df_1 (pd.DataFrame): The generated data for table 1.
            df_2 (pd.DataFrame): The generated data for table 2.
        """
        df_1_synth = self._synthesizers[0].synthesize(num_rows, produce_nans=produce_nans)
        keys, fk_counts = df_1_synth[self.keys[0]], df_1_synth.pop('fk_count')

        _df_2_synth = []
        for fk, counts in zip(keys, fk_counts):
            if counts > 0:
                df_synth = self._synthesizers[1].synthesize(counts)
                df_synth[self.relation[self.keys[0]]] = fk
                _df_2_synth.append(df_synth)

        df_2_synth = pd.concat(_df_2_synth, ignore_index=True)

        pk_model = self._synthesizers[1]._df_model_independent[self.keys[1]]
        df_2_synth[self.keys[1]] = pk_model.sample(len(df_2_synth))

        return df_1_synth, df_2_synth

    def _build_relation(self, df: Tuple[pd.DataFrame, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Augment table 1 with the value counts of each unique foreign key in table 2. The foreign key is also dropped
        from table 2, as this is inserted from table 1 to maintain the referential integrity.
        """

        df_1, df_2 = df
        fk_name = self.relation[self.keys[0]]
        fk_counts = _get_foreign_key_count(df_2, fk_name)

        df_1 = _join_foreign_key_count(df_1, self.keys[0], fk_counts)
        df_2 = df_2.drop(columns=fk_name)

        return df_1, df_2

    # alias method for learn
    fit = learn

    # alias method for synthesize
    sample = synthesize


def _get_foreign_key_count(df: pd.DataFrame, key: str) -> pd.DataFrame:
    fk_count = df.value_counts(key)
    fk_count.name = 'fk_count'
    return fk_count


def _join_foreign_key_count(df: pd.DataFrame, key: str, fk_count: pd.Series) -> pd.DataFrame:
    df = df.set_index(key).join(fk_count, how='left')
    df[fk_count.name] = df[fk_count.name].fillna(0).astype(int)
    return df.reset_index()
