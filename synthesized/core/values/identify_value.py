from math import log, sqrt

from .address import AddressValue
from .categorical import CategoricalValue
from .continuous import ContinuousValue
from .date import DateValue
from .enumeration import EnumerationValue
from .gaussian import GaussianValue
from .identifier import IdentifierValue
from .person import PersonValue
from .powerlaw import PowerlawValue
from .probability import ProbabilityValue
from .sampling import SamplingValue
from .gumbel import GumbelDistrValue
from .gamma import GammaDistrValue
from .weibull import WeibullDistrValue

from scipy.stats import ks_2samp
from scipy.stats import gamma
from scipy.stats import gumbel_r
from scipy.stats import weibull_min

MIN_FIT_DISTANCE = 0.1
CONT_DISTRIBUTIONS = [gamma, gumbel_r, weibull_min]
DIST_TO_VALUE_MAPPING = {
    'gamma': GammaDistrValue,
    'gumbel_r': GumbelDistrValue,
    'weibull_min': WeibullDistrValue
}


def identify_value(module, name, dtype, data):

    if name in (getattr(module, 'gender_label', None), getattr(module, 'name_label', None), getattr(module, 'firstname_label', None), getattr(module, 'lastname_label', None), getattr(module, 'email_label', None)):
        if module.person_value is None:
            value = module.add_module(
                module=PersonValue, name='person', gender_label=module.gender_label,
                name_label=module.name_label, firstname_label=module.firstname_label,
                lastname_label=module.lastname_label, email_label=module.email_label
            )
            module.person_value = value

    elif name in (getattr(module, 'postcode_label', None), getattr(module, 'street_label', None)):
        if module.address_value is None:
            value = module.add_module(
                module=AddressValue, name='address', postcode_level=1,
                postcode_label=module.postcode_label, street_label=module.street_label
            )
            module.address_value = value

    elif name == getattr(module, 'identifier_label', None):
        value = module.add_module(
            module=IdentifierValue, name=name, embedding_size=None
        )
        module.identifier_value = value

    elif dtype.kind == 'M':  # 'm' timedelta
        if module.date_value is not None:
            raise NotImplementedError
        value = module.add_module(module=DateValue, name=name)
        module.date_value = value

    elif dtype.kind == 'b':
        value = module.add_module(
            module=CategoricalValue, name=name, categories=[False, True], capacity=module.capacity
        )

    elif dtype.kind == 'O' and hasattr(dtype, 'categories'):
        value = module.add_module(
            module=CategoricalValue, name=name, categories=dtype.categories,
            capacity=module.capacity, pandas_category=True
        )

    else:
        num_data = len(data)
        num_unique = data[name].nunique()

        if num_unique <= 2.5 * log(num_data):
            value = module.add_module(module=CategoricalValue, name=name, capacity=module.capacity)

        elif dtype.kind == 'f' and (data[name] <= 1.0).all() and (data[name] >= 0.0).all():
            value = module.add_module(module=ProbabilityValue, name=name)

        elif dtype.kind == 'f':
            distance = 1
            for distr in CONT_DISTRIBUTIONS:
                params = distr.fit(data[name])
                ghost_sample = distr.rvs(*params, size=len(data[name]))
                distance_distr = ks_2samp(data[name], ghost_sample)[0]
                if distance_distr < distance:
                    distance = distance_distr
                    distr_fitted = [distr, params]
            if distance < MIN_FIT_DISTANCE:
                value = module.add_module(
                    module=DIST_TO_VALUE_MAPPING[distr_fitted[0].name], name=name, params=distr_fitted[1]
                )
            else:
                # default continuous value fit
                value = module.add_module(
                    module=ContinuousValue, name=name, positive=(data[name] > 0.0).all(),
                    nonnegative=(data[name] >= 0.0).all()
                )

        elif False:
            value = module.add_module(module=GaussianValue, name=name)

        elif False:
            value = module.add_module(module=PowerlawValue, name=name)

        elif dtype.kind == 'i':
            # positive since implicit floor
            value = module.add_module(
                module=ContinuousValue, name=name, positive=(data[name] >= 0).all(), integer=True
            )

        elif num_unique <= sqrt(num_data):
            value = module.add_module(
                module=CategoricalValue, name=name, capacity=module.capacity, similarity_based=True
            )

        elif dtype.kind != 'f' and num_unique == num_data and data[name].is_monotonic:
            value = module.add_module(module=EnumerationValue, name=name)

        else:
            value = module.add_module(module=SamplingValue, name=name)
            print(name, dtype, num_data, num_unique)
            raise NotImplementedError

    return value
