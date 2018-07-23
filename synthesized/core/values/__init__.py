from .categorical import CategoricalValue
from .continuous import ContinuousValue
from .identifier import IdentifierValue
from .value import Value

value_modules = dict(
    categorical=CategoricalValue,
    continuous=ContinuousValue,
    identifier=IdentifierValue
)
