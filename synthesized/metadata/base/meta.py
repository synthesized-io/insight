from contextlib import contextmanager
from typing import Any, Dict, Generic, Iterator, List, Mapping, Optional, Sequence, Type, TypeVar, cast  # noqa

import pandas as pd

from ...util import get_all_subclasses

MetaType = TypeVar('MetaType', bound='Meta', covariant=True)
MetaTypeArg = TypeVar('MetaTypeArg', bound='Meta')


class Meta(Mapping[str, MetaType], Generic[MetaType]):
    """
    Base class for meta information that describes a dataset.

    The generic MetaType parameter defines the type of the children metas.

    Implements a hierarchical tree structure to describe arbitrary nested
    relations between data. Instances of Meta act as root nodes in the tree,
    and each branch can lead to another Meta or a leaf node, see ValueMeta.

    Args:
        name: a descriptive name.
        children (Sequence[MetaType]): the children owned by this meta. This is immutable
            and must be defined during initialization.

    Examples:
        Custom nested structures can be easily built with this class, for example:

        >>> customer = Meta('customer', children=[Meta('name'), Meta('age')])

        Meta objects are iterable, and allow iterating through the
        children:

        >>> for child_meta in customer:
        >>>     print(child_meta)
    """
    class_name: str = 'Meta'

    def __init__(self, name: str, children: Optional[Sequence[MetaType]] = None, num_rows: Optional[int] = None):
        self.name = name
        self.num_rows = num_rows
        self._children: Dict[str, MetaType] = {c.name: c for c in children} if children is not None else dict()
        self._extracted: bool = False

    @property
    def children(self) -> Sequence[MetaType]:
        """Return the children of this Meta."""
        return [child for child in self.values()]

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={self.name})'

    def extract(self, df: pd.DataFrame) -> 'Meta':
        """Extract the children of this Meta."""
        with self.unfold(df) as sub_df:
            for child in self.children:
                child.extract(sub_df)
        self.num_rows = len(df)
        self._extracted = True
        return self

    def update_meta(self, df: pd.DataFrame) -> 'Meta':
        """update the children of this Meta."""
        with self.unfold(df) as sub_df:
            for child in self.children:
                child.update_meta(sub_df)
        self.num_rows = self.num_rows + len(df) if self.num_rows is not None else len(df)
        self._extracted = True
        return self

    def convert_df_for_children(self, df):
        """Expands the dataframe to contain the columns of the metas children."""
        pass

    def revert_df_from_children(self, df):
        """Collapses the dataframe to no longer contain the meta's children columns."""
        pass

    @contextmanager
    def unfold(self, df: pd.DataFrame) -> pd.DataFrame:
        columns = df.columns
        self.convert_df_for_children(df)
        yield df
        self.revert_df_from_children(df)
        for col in columns:  # set the original column order
            sr = df[col]
            df.drop(col, axis=1, inplace=True)
            df[col] = sr

    def __getitem__(self, k: str) -> MetaType:
        return self._children[k]

    def __iter__(self) -> Iterator[str]:
        for key in self._children:
            yield key

    def __len__(self) -> int:
        return len(self._children)

    def to_dict(self) -> Dict[str, object]:
        """
        Convert the Meta to a dictionary.

        The tree structure is converted to the following form:

        .. code-block:: python

            {
                attr: value,
                children: {
                    name: {**value_meta_attr.__dict__}
                }
            }

        See also:

            Meta.from_dict: construct a Meta from a dictionary
        """
        d = {
            "name": self.name,
            "class_name": self.__class__.__name__,
            "extracted": self._extracted,
            "num_rows": self.num_rows
        }

        if len(self.children) > 0:
            d['children'] = {child.name: child.to_dict() for child in self.children}
        else:
            d['children'] = None

        return d

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> 'Meta':
        """
        Construct a Meta from a dictionary.

        See example in Meta.to_dict() for the required structure.

        See also:

            Meta.to_dict: convert a Meta to a dictionary
        """
        name = cast(str, d["name"])
        extracted = cast(bool, d["extracted"])
        num_rows = cast(Optional[int], d["num_rows"])
        children = cls.children_from_dict(d)

        meta = cls(name=name, num_rows=num_rows, children=children)
        meta._extracted = extracted

        return meta

    @classmethod
    def children_from_dict(cls, d: Dict[str, Any]) -> Optional[Sequence[MetaType]]:
        if d["children"] is None:
            return None

        children_dict = cast(Dict[str, Dict[str, object]], d["children"])
        meta_children: List[MetaType] = []
        for child in children_dict.values():
            class_name = cast(str, child['class_name'])
            meta_children.append(Meta.from_name_and_dict(class_name, child))  # type: ignore

        return meta_children

    @classmethod
    def from_name_and_dict(cls: Type[MetaTypeArg], class_name: str, d: Dict[str, object]) -> MetaTypeArg:
        """
        Construct a Meta from a meta class name and a dictionary.

        See also:

            Meta.from_dict: construct a Meta from a dictionary
        """

        registy = cls.get_registry()
        if class_name not in registy.keys():
            raise ValueError(f"Given meta {class_name} not found in Meta subclasses.")

        return registy[class_name].from_dict(d)

    @classmethod
    def get_registry(cls: Type[MetaType]) -> Dict[str, Type[MetaType]]:
        return {sc.__name__: sc for sc in get_all_subclasses(cls).union({cls})}

    def __eq__(self, other) -> bool:
        return {k: v for k, v in vars(self).items()
                if v is not self and not k.startswith('__')} == {k: v for k, v in vars(other).items()
                                                                 if v is not other and not k.startswith('__')}
