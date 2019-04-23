from .address import AddressValue
from .compound_address import CompoundAddressValue
from .categorical import CategoricalValue
from .continuous import ContinuousValue
from .date import DateValue
from .enumeration import EnumerationValue
from .identifier import IdentifierValue
from .identify_value import identify_value
from .nan import NanValue
from .person import PersonValue
from .probability import ProbabilityValue
from .sampling import SamplingValue
from .value import Value


value_modules = dict(
    address=AddressValue,
    categorical=CategoricalValue,
    continuous=ContinuousValue,
    date=DateValue,
    enumeration=EnumerationValue,
    identifier=IdentifierValue,
    person=PersonValue,
    probability=ProbabilityValue,
    sampling=SamplingValue
)
