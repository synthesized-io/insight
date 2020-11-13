from typing import List, Type

from .contains_nans import ContainsNans
from .interval import Interval
from .is_integer import IsInteger
from .validity_rule import ValidityRule


NumericalRules: List[Type[ValidityRule]] = [
    Interval,
    IsInteger,
    ContainsNans
]

CategoricalRules = [
    ContainsNans
]

__all__ = [
    'NumericalRules', 'ContainsNans',
    'ValidityRule', 'Interval', 'ContainsNans'
]
