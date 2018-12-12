from .address import AddressValue
from .categorical import CategoricalValue
from .continuous import ContinuousValue
from .date import DateValue
from .enumeration import EnumerationValue
from .gaussian import GaussianValue
from .identifier import IdentifierValue
from .identify_value import identify_value
from .person import PersonValue
from .powerlaw import PowerlawValue
from .probability import ProbabilityValue
from .sampling import SamplingValue
from .beta import BetaDistrValue
from .gamma import GammaDistrValue
from .gumbel import GumbelDistrValue
from .weibull import WeibullDistrValue
from .value import Value


value_modules = dict(
    address=AddressValue,
    categorical=CategoricalValue,
    continuous=ContinuousValue,
    date=DateValue,
    enumeration=EnumerationValue,
    gaussian=GaussianValue,
    identifier=IdentifierValue,
    person=PersonValue,
    powerlaw=PowerlawValue,
    orobability=ProbabilityValue,
    beta=BetaDistrValue,
    gamma=GammaDistrValue,
    gumbel_r=GumbelDistrValue,
    weibull_min=WeibullDistrValue,
    probability=ProbabilityValue,
    sampling=SamplingValue
)

