from typing import Dict, List, MutableMapping, Optional

import pandas as pd

from .base import Meta


class DataFrameMeta(Meta, MutableMapping[str, 'Meta']):
    """
    Meta to describe an arbitrary data frame.

    Each column is described by a derived ValueMeta object.

    Attributes:
        id_index: NotImplemented
        time_index: NotImplemented
        column_aliases: dictionary mapping column names to an alias.
        annotations: A list of the metas' names in the DF that have been annotated.
    """
    def __init__(
            self, name: str, id_index: Optional[str] = None, time_index: Optional[str] = None,
            column_aliases: Optional[Dict[str, str]] = None, num_columns: Optional[int] = None,
            num_rows: Optional[int] = None, annotations: Optional[List[str]] = None
    ):
        # TODO: This init should receive children as an optional argument so that id_index, time_index,
        #       column_aliases, and annotations can all be checked to be consistant.
        super().__init__(name=name)
        self.id_index = id_index
        self.time_index = time_index
        self.column_aliases = column_aliases if column_aliases is not None else {}
        self.columns = None
        self.num_columns = num_columns
        self.num_rows = num_rows
        self.annotations = annotations if annotations is not None else []

    def extract(self, df: pd.DataFrame) -> 'DataFrameMeta':
        super().extract(df)
        self.columns = df.columns
        self.num_columns = len(df.columns)
        self.num_rows = len(df)
        return self

    def convert_df_for_children(self, df: pd.DataFrame):
        for ann in self.annotations:
            self[ann].revert_df_from_children(df)

    def revert_df_from_children(self, df: pd.DataFrame):
        for ann in self.annotations:
            self[ann].convert_df_for_children(df)

    def annotate(self, annotation: Meta):
        if annotation.name in self.annotations:
            raise ValueError(f"Annotation {annotation} already applied.")

        # Make sure that the annotation's children are the same as the ones in the dataframe.
        # TODO: Whilst this guarantees correctly named children, the types aren't necessarily correct.
        #       Also, setting the children of an annotation technically shouldn't be allowed (not mutable).
        annotation.children = [self.pop(child) for child in annotation]
        self[annotation.name] = annotation

        self.annotations.append(annotation.name)

    def unannotate(self, annotation: Meta):
        if annotation.name not in self.annotations:
            raise ValueError(f"Annotation {annotation} cannot be stripped as it isn't in the DF Meta.")

        # TODO: This may create a state in the df_meta different to just extracting the df with no
        #       annotations. See the todo in `DataFrameMeta.annotate`.
        for name, child in annotation.items():
            self[name] = child

        self.pop(annotation.name)
        self.annotations.remove(annotation.name)

    def __setitem__(self, k: str, v: Meta) -> None:
        self._children[k] = v

    def __delitem__(self, k: str) -> None:
        del self._children[k]

    def to_dict(self) -> Dict[str, object]:
        d = super().to_dict()
        d.update({
            "id_index": self.id_index,
            "time_index": self.time_index,
            "column_aliases": self.column_aliases,
            "num_columns": self.num_columns,
            "num_rows": self.num_rows,
            "annotations": self.annotations
        })

        return d
