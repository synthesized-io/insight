from typing import List, Dict, Type, TypeVar, MutableMapping, Iterator, cast

import pandas as pd


MetaType = TypeVar('MetaType', bound='Meta')


class Meta(MutableMapping[str, 'Meta']):
    """
    Base class for meta information that describes a dataset.

    Implements a hierarchical tree structure to describe arbitrary nested
    relations between data. Instances of Meta act as root nodes in the tree,
    and each branch can lead to another Meta or a leaf node, see ValueMeta.

    Attributes:
        name: a descriptive name.

    Examples:
        Custom nested structures can be easily built with this class, for example:

        >>> customer = Meta('customer')

        Meta objects are iterable, and allow iterating through the
        children:

        >>> for child_meta in customer:
        >>>     print(child_meta)
    """
    STR_TO_META: Dict[str, Type['Meta']] = {}
    class_name: str = 'Meta'

    def __init__(self, name: str):
        self.name = name
        self._children: Dict[str, 'Meta'] = dict()
        self._extracted: bool = False

    def __init_subclass__(cls: Type[MetaType]) -> None:
        super().__init_subclass__()
        Meta.STR_TO_META[cls.class_name] = cls

    @property
    def children(self) -> List['Meta']:
        """Return the children of this Meta."""
        return [child for child in self.values()]

    @children.setter
    def children(self, children: List['Meta']) -> None:
        self._children = {child.name: child for child in children}

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={self.name})'

    def extract(self, df: pd.DataFrame) -> 'Meta':
        """Extract the children of this Meta."""
        for child in self.children:
            child.extract(df)
        self._extracted = True
        return self

    def __getitem__(self, k: str) -> 'Meta':
        return self._children[k]

    def __setitem__(self, k: str, v: 'Meta') -> None:
        self._children[k] = v

    def __delitem__(self, k: str) -> None:
        del self._children[k]

    def __iter__(self) -> Iterator[str]:
        for key in self._children:
            yield key

    def __len__(self) -> int:
        return len(self._children)

    def to_dict(self) -> Dict[str, object]:
        """
        Convert the Meta to a dictionary.
        The tree structure is converted to the following form:
        Meta.__class.__.__name__: {
            attr: value,
            value_meta_attr: {
                value_meta_attr.__class__.__name__: (**value_meta_attr.__dict__}
                )
            }
        }
        Examples:
            >>> customer = Meta('customer')
            >>> customer['title'] = Nominal('title')
            >>> customer['address'] = Meta('customer_address')
            >>> customer['address']['street'] = Nominal('street')
            Convert to dictionary:
            >>> customer.to_dict()
            {
                'name': 'customer',
                'class': 'Meta',
                'extracted': False,
                'children': {
                    'age': {
                        'name': 'age',
                        'class': 'Integer',
                        'extracted': False,
                        'dtype': 'int64',
                        'domain': [],
                        'probabilities': [],
                        'similarity_based': True,
                        'min': None,
                        'max': None,
                        'distribution': None,
                        'monotonic': False,
                        'nonnegative': None
                        }
                    },
                    'customer_address': {
                        'name': 'customer_address',
                        'class': 'Meta',
                        'extracted': False,
                        'children': {
                            'street': {
                                'name': 'street',
                                'class': 'Nominal',
                                'extracted': False,
                                'dtype': None,
                                'domain': [],
                                'probabilities': []
                            }
                        }
                    }
                }
            }
        See also:
            Meta.from_dict: construct a Meta from a dictionary
        """
        d = {
            "name": self.name,
            "class_name": self.class_name,
            "extracted": self._extracted
        }

        if len(self.children) > 0:
            d['children'] = {child.name: child.to_dict() for child in self.children}

        return d

    @classmethod
    def from_dict(cls: Type['MetaType'], d: Dict[str, object]) -> 'MetaType':
        """
        Construct a Meta from a dictionary.
        See example in Meta.to_dict() for the required structure.
        See also:
            Meta.to_dict: convert a Meta to a dictionary
        """
        name = cast(str, d["name"])
        d.pop("class")

        extracted = d.pop("extracted")
        children = cast(Dict[str, Dict[str, object]], d.pop("children")) if "children" in d else None

        meta = cls(name=name)
        for attr, value in d.items():
            setattr(meta, attr, value)

        setattr(meta, '_extracted', extracted)

        if children is not None:
            meta_children = []
            for child in children.values():
                class_name = cast(str, child['class_name'])
                meta_children.append(Meta.STR_TO_META[class_name].from_dict(child))

            meta.children = meta_children

        return meta
