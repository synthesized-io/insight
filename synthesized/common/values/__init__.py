from .address import AddressValue
from .categorical import CategoricalValue
from .compound_address import CompoundAddressValue
from .continuous import ContinuousValue
from .date import DateValue
from .enumeration import EnumerationValue
from .identifier import IdentifierValue
from .identify_rules import identify_rules
from .identify_value import identify_value
from .nan import NanValue
from .person import PersonValue
from .probability import ProbabilityValue
from .rule import RuleValue
from .sampling import SamplingValue
from .value import Value
from ..module import register


register(name='address', module=AddressValue)
register(name='categorical', module=CategoricalValue)
register(name='compound_address', module=CompoundAddressValue)
register(name='continuous', module=ContinuousValue)
register(name='date', module=DateValue)
register(name='enumeration', module=EnumerationValue)
register(name='identifier', module=IdentifierValue)
register(name='nan', module=NanValue)
register(name='person', module=PersonValue)
register(name='probability', module=ProbabilityValue)
register(name='rule', module=RuleValue)
register(name='sampling', module=SamplingValue)


__all__ = ['Value', 'identify_rules', 'identify_value']