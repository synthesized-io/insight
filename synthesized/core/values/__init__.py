from .address import AddressValue
from .categorical import CategoricalValue
from .continuous import ContinuousValue
from .date import DateValue
from .identifier import IdentifierValue
from .person import PersonValue
from .value import Value


value_modules = dict(
    address=AddressValue,
    categorical=CategoricalValue,
    continuous=ContinuousValue,
    date=DateValue,
    identifier=IdentifierValue,
    person=PersonValue
)
