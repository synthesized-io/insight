from .categorical import CategoricalValue
from .continuous import ContinuousValue
from .date import DateValue
from .identifier import IdentifierValue
from .value import Value

value_modules = dict(
    categorical=CategoricalValue,
    continuous=ContinuousValue,
    date=DateValue,
    identifier=IdentifierValue
)
