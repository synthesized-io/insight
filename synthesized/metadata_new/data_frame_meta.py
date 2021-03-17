from itertools import chain
from typing import Dict, List, MutableMapping, Optional, Sequence

import pandas as pd

from .base import Meta, ValueMeta


class DataFrameMeta(Meta[Meta], MutableMapping[str, Meta]):
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
            self, name: str, children: Optional[Sequence[Meta]] = None, id_index: Optional[str] = None,
            time_index: Optional[str] = None, column_aliases: Optional[Dict[str, str]] = None,
            columns: Optional[List[str]] = None,
            num_columns: Optional[int] = None, num_rows: Optional[int] = None, annotations: Optional[List[str]] = None
    ):
        child_names = [c.name for c in children] if children is not None else []
        annotations_list = annotations if annotations is not None else []
        column_aliases_dict = column_aliases if column_aliases is not None else {}

        if any([
            n in child_names
            for n in chain([id_index, time_index], annotations_list, column_aliases_dict)
            if n is not None
        ]):
            raise ValueError("Children metas don't match the given indices/aliases/annotations")

        super().__init__(name=name, children=children)
        self.id_index = id_index
        self.time_index = time_index
        self.column_aliases = column_aliases_dict
        self.columns = columns
        self.num_columns = num_columns
        self.num_rows = num_rows
        self.annotations = annotations_list

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

    def annotate(self, annotation: ValueMeta):
        if annotation.name in self.annotations:
            raise ValueError(f"Annotation {annotation} already applied.")

        # Make sure the children in the annotation are the same as in the dataframe
        ann_children = [self.pop(child_name) for child_name in annotation]
        if not all([c1 is c2 for c1, c2 in zip(annotation.children, ann_children)]):
            raise ValueError(f"The children of Annotation {annotation} aren't in this dataframe.")
        self[annotation.name] = annotation

        self.annotations.append(annotation.name)

    def unannotate(self, annotation: ValueMeta):
        if annotation.name not in self.annotations:
            raise ValueError(f"Annotation {annotation} cannot be stripped as it isn't in the DF Meta.")

        for name, child in annotation.items():
            self[name] = child

        self.pop(annotation.name)

    @property
    def children(self) -> Sequence[Meta]:
        return [child for child in self.values()]

    @children.setter
    def children(self, children: Sequence[Meta]) -> None:
        self._children = {child.name: child for child in children}

    def __setitem__(self, k: str, v: Meta) -> None:
        self._children[k] = v

    def __delitem__(self, k: str) -> None:
        del self._children[k]
        if k in self.annotations:
            self.annotations.remove(k)

    def to_dict(self) -> Dict[str, object]:
        d = super().to_dict()
        d.update({
            "id_index": self.id_index,
            "time_index": self.time_index,
            "column_aliases": self.column_aliases,
            "columns": self.columns,
            "num_columns": self.num_columns,
            "num_rows": self.num_rows,
            "annotations": self.annotations
        })

        return d

    def copy(self) -> 'DataFrameMeta':
        """Returns a shallow copy of the data frame meta."""
        df_meta = DataFrameMeta(
            name=self.name, id_index=self.id_index, time_index=self.time_index,
            column_aliases=self.column_aliases.copy(), num_columns=self.num_columns, num_rows=self.num_rows,
            annotations=self.annotations.copy(), columns=self.columns.copy() if self.columns is not None else None
        )
        df_meta.update(self)
        return df_meta
