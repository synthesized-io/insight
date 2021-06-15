from typing import List, Sequence, Union

import numpy as np
import pandas as pd

from .base import Unifier
from ..highdim import HighDimSynthesizer
from ...metadata import DataFrameMeta
from ...model import DataFrameModel, DiscreteModel


class EnsembleUnifier(Unifier):
    """
    Data Oracle unifier that uses an ensemble of HighDimSynthesizers to unify columns across dataframes.

    Attributes:
        df_metas: A list of all the DataFrameMeta objects for the DataFrames that are added to the update method.
        ensemble: A list of the trained HighDimSynthesizer objects for each DataFrame, used in generation.
    """

    def __init__(self, ensemble: Sequence[HighDimSynthesizer] = None):
        self.ensemble: List[HighDimSynthesizer] = list(ensemble) if ensemble is not None else []

    def update(self,
               dfs: Union[pd.DataFrame, Sequence[pd.DataFrame]] = None,
               df_metas: Union[DataFrameMeta, Sequence[DataFrameMeta]] = None,
               num_iterations: int = None) -> None:
        """
        Incorporates the DataFrame and DataFrameMeta object into the unifier for generation.

        Trains a single HighDimSynthesizer object for a new dataframe.

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
                for df, df_meta in zip(dfs, df_metas):
                    synthesizer = HighDimSynthesizer(df_meta)
                    synthesizer.learn(df, num_iterations=num_iterations)
                    self.ensemble.append(synthesizer)
            else:
                raise ValueError("length of dfs and df_metas provided don't match")

    def query(self, columns: Sequence[str], num_rows: int, query_iterations: int = 20) -> pd.DataFrame:
        """
        Queries the unifier to request a dataframe with the specified columns.

        Generates from all HighDimSynthesizers,
        tries to match rows in the all the DataFrames to produce a unified output.

        Args:
            columns: the list of columns to generate from.
            num_rows: number of rows of the dataframe to generate.
            query_iterations:
                number of attempts at generating num_rows and unifying before
                giving up and returning a DataFrame with fewer rows than requested.
        """
        if not set(columns) <= {col for synthesizer in self.ensemble for col in synthesizer.df_meta}:
            raise ValueError("Column requested is not one that has been entered in an update call")
        unified_df = pd.DataFrame(columns=columns)

        for _ in range(query_iterations):
            if len(unified_df) > num_rows:
                break
            ensemble_dfs = [synthesizer.synthesize(num_rows) for synthesizer in self.ensemble]
            df = ensemble_dfs[0]
            relevant_cols = set(columns).intersection(df.columns)
            df = df[relevant_cols]

            for synthesizer, df_ in zip(self.ensemble[1:], ensemble_dfs[1:]):
                relevant_cols = set(columns).intersection(df_.columns)
                df_ = df_[relevant_cols]
                df = _resolve_overlap(df, df_, synthesizer.df_model)

            unified_df = unified_df.append(df, ignore_index=True)
        return unified_df.iloc[:num_rows]


def _resolve_overlap(df1: pd.DataFrame, df2: pd.DataFrame, df_model: DataFrameModel):
    """
    Core function for unifying two dataframes through matching.

    Takes two dataframes, detects overlaps and finds rows where the overlaps are very similar.
    Then joins these rows together and returns them in a dataframe. To determine similarity the
    df_model is used to determine the type of each overlapping column.
    If categorical: elements of the DataFrame are considered similar if the values exactly match.
    If continuous: elements are considered similar if they lie within 0.1 of one another.
        TODO: change continuous comparison to take into account the scale of the data.

    Args:
        df: first DataFrame to unify.
        df2: second DataFrame to unify.
        df_model: DataFrameModel that determines how unification is done,
    """
    output_df = pd.DataFrame(columns=df1.columns.union(df2.columns))
    overlap = df1.columns.intersection(df2.columns)

    for _, row in df1.iterrows():
        similar_rows_bool = np.ones(shape=(len(df2),))
        for column in overlap:
            if isinstance(df_model[column], DiscreteModel):
                similar_rows_bool *= row[column] == df2[column]
            else:
                similar_rows_bool *= np.isclose(row[column], df2[column], atol=0, rtol=0.1)

        similar_rows = df2[similar_rows_bool.astype(bool)]
        if len(similar_rows) > 0:
            output_df = output_df.append(row.combine_first(similar_rows.iloc[0, :]))
            df2 = df2.drop(index=similar_rows.index[0])

    # convert data back to original dtypes
    for col in df1.columns.difference(df2.columns):
        output_df[col] = output_df[col].astype(df1[col].dtype, copy=False)

    for col in df2.columns.difference(df1.columns):
        output_df[col] = output_df[col].astype(df2[col].dtype, copy=False)

    # TODO: for overlap, some logic could be added to account for when dtypes are different
    for col in overlap:
        output_df[col] = output_df[col].astype(df1[col].dtype, copy=False)

    return output_df.reset_index(drop=True)
