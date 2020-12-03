from abc import abstractmethod
from typing import Generic, TypeVar, List, Dict, Type, Optional, Any, MutableMapping, cast

import numpy as np

KT = TypeVar('KT')
OType = TypeVar("OType", np.datetime64, np.timedelta64, int, float, bool, covariant=True)
DomainType = TypeVar('DomainType', bound='Domain[Any]')


class Domain(Generic[KT]):
    STR_TO_DOMAIN: Dict[str, Type['Domain[Any]']] = {}
    domain_type: str = 'Domain'

    def __init_subclass__(cls: Type['Domain[Any]']) -> None:
        super().__init_subclass__()
        Domain.STR_TO_DOMAIN[cls.domain_type] = cls

    @abstractmethod
    def __contains__(self, item: KT) -> bool:
        pass

    def to_dict(self) -> MutableMapping[str, object]:
        d = {
            "domain_type": self.domain_type
        }
        return cast(MutableMapping[str, object], d)

    @classmethod
    def from_dict(cls: Type['DomainType'], d: MutableMapping[str, object]) -> 'DomainType':
        """
        Construct a Meta from a dictionary.
        See example in Meta.to_dict() for the required structure.
        See also:
            Meta.to_dict: convert a Meta to a dictionary
        """
        if "domain_type" in d:
            d.pop("domain_type")

        domain = cls()
        for attr, value in d.items():
            setattr(domain, attr, value)

        return domain


class CategoricalDomain(Domain[KT], Generic[KT]):
    domain_type = "CategoricalDomain"

    def __init__(self, categories: List[KT]):
        self.categories = categories

    def __contains__(self, item: KT) -> bool:
        return item in self.categories

    def to_dict(self) -> MutableMapping[str, object]:
        d = super().to_dict()
        d.update({
            "categories": self.categories
        })
        return d


class OrderedDomain(Domain[OType], Generic[OType]):
    domain_type = "OrderedDomain"

    def __init__(self, min: Optional[OType] = None, max: Optional[OType] = None):
        self.min: Optional[OType] = min
        self.max: Optional[OType] = max

    def __contains__(self, item: OType) -> bool:
        return (self.min is None or self.min <= item) and (self.max is None or item <= self.max)

    def to_dict(self) -> MutableMapping[str, object]:
        d = super().to_dict()
        d.update({
            "min": self.min,
            "max": self.max
        })
        return d
