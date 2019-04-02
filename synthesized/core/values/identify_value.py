from math import log, sqrt

import pandas as pd

from .address import AddressValue
from .compound_address import CompoundAddressValue
from .categorical import CategoricalValue
from .continuous import ContinuousValue
from .date import DateValue
from .enumeration import EnumerationValue
from .identifier import IdentifierValue
from .nan import NanValue
from .person import PersonValue
from .sampling import SamplingValue
from .gumbel import GumbelDistrValue
from .gilbrat import GilbratDistrValue
from .gamma import GammaDistrValue
from .weibull import WeibullDistrValue
from .uniform import UniformDistrValue
from .lognorm import LognormDistrValue

from scipy.stats import kstest, gamma, gumbel_r, weibull_min, gilbrat, uniform, norm, lognorm

REMOVE_OUTLIERS_PCT = 1.0
MAX_FIT_DISTANCE = 1.0
MIN_FIT_DISTANCE = 0.15
CONT_DISTRIBUTIONS = [uniform, gamma, gumbel_r, weibull_min, gilbrat, lognorm]
DIST_TO_VALUE_MAPPING = {
    'uniform': UniformDistrValue,
    'gamma': GammaDistrValue,
    'lognorm': LognormDistrValue,
    'gumbel_r': GumbelDistrValue,
    'weibull_min': WeibullDistrValue,
    'gilbrat': GilbratDistrValue
}


def identify_value(module, name, dtype, data):
    value = None

    if name in (getattr(module, 'gender_label', None), getattr(module, 'name_label', None), getattr(module, 'firstname_label', None), getattr(module, 'lastname_label', None), getattr(module, 'email_label', None)):
        if module.person_value is None:
            value = module.add_module(
                module=PersonValue, name='person', gender_label=module.gender_label,
                name_label=module.name_label, firstname_label=module.firstname_label,
                lastname_label=module.lastname_label, email_label=module.email_label
            )
            module.person_value = value
        return value

    elif name in (getattr(module, 'postcode_label', None), getattr(module, 'street_label', None)):
        if module.address_value is None:
            value = module.add_module(
                module=AddressValue, name='address', postcode_level=1,
                postcode_label=module.postcode_label, street_label=module.street_label,
                capacity=module.capacity
            )
            module.address_value = value
        return value

    elif name == getattr(module, 'address_label', None):
        value = module.add_module(
            module=CompoundAddressValue, name='address', postcode_level=1,
            address_label=module.address_label,
            postcode_regex=module.postcode_regex,
            capacity=module.capacity
        )
        module.address_value = value
        return value

    elif name == getattr(module, 'identifier_label', None):
        value = module.add_module(module=IdentifierValue, name=name, capacity=module.capacity)
        module.identifier_value = value
        return value

    to_numeric = False
    if dtype.kind == 'O':
        numeric_data = pd.to_numeric(data[name], errors='coerce')
        if numeric_data.isna().sum() / len(numeric_data) < 0.25:
            data[name] = numeric_data
            dtype = numeric_data.dtype
            assert dtype.kind in ('f', 'i')
            to_numeric = True

    is_nan = data[name].isna().any()
    clean = data.dropna(subset=(name,))

    if dtype.kind == 'M':  # 'm' timedelta
        # if module.date_value is not None:
        #     raise NotImplementedError
        value = module.add_module(module=DateValue, name=name, capacity=module.capacity)
        # module.date_value = value

    elif dtype.kind == 'b':
        assert not is_nan
        value = module.add_module(
            module=CategoricalValue, name=name, categories=[False, True], capacity=module.capacity
        )

    elif dtype.kind == 'O':
        if hasattr(dtype, 'categories'):
            value = module.add_module(
                module=CategoricalValue, name=name, categories=dtype.categories,
                capacity=module.capacity, pandas_category=True
            )
            is_nan = False

        else:
            try:
                pd.to_datetime(clean)
                # if module.date_value is not None:
                #     raise NotImplementedError
                value = module.add_module(module=DateValue, name=name, capacity=module.capacity)
                # module.date_value = value
            except ValueError:
                pass

    num_data = len(clean)
    num_unique = clean[name].nunique()

    if value is None and num_data > 0:
        if num_unique <= 2.5 * log(num_data):
            assert not to_numeric
            value = module.add_module(module=CategoricalValue, name=name, capacity=module.capacity)
            is_nan = False

        # elif dtype.kind == 'f' and (clean[name] <= 1.0).all() and (clean[name] >= 0.0).all():
        #     value = module.add_module(module=ProbabilityValue, name=name)

        elif dtype.kind == 'f' or dtype.kind == 'i':
            min_distance = MAX_FIT_DISTANCE
            column_cleaned = ContinuousValue.remove_outliers(clean, name, pct=REMOVE_OUTLIERS_PCT)[name]
            for distr in CONT_DISTRIBUTIONS:
                params = distr.fit(column_cleaned)
                transformed = norm.ppf(distr.cdf(column_cleaned, *params))
                norm_dist, _ = kstest(transformed, 'norm')
                if norm_dist < min_distance:
                    min_distance = norm_dist
                    distr_fitted, params_fitted = distr, params
            if min_distance < MIN_FIT_DISTANCE:
                value = module.add_module(
                    module=DIST_TO_VALUE_MAPPING[distr_fitted.name], name=name, integer=dtype.kind == 'i', to_numeric=to_numeric, params=params_fitted
                )
            elif dtype.kind == 'f':
                # default continuous value fit
                value = module.add_module(
                    module=ContinuousValue, name=name, positive=(clean[name] > 0.0).all(),
                    nonnegative=(clean[name] >= 0.0).all(), to_numeric=to_numeric
                )
            elif dtype.kind == 'i':
                value = module.add_module(
                    module=ContinuousValue, name=name, positive=(clean[name] >= 0).all(),
                    integer=True, to_numeric=to_numeric
                )

        elif num_unique <= sqrt(num_data):
            assert not to_numeric
            value = module.add_module(
                module=CategoricalValue, name=name, capacity=module.capacity, similarity_based=True
            )
            is_nan = False

        elif dtype.kind != 'f' and num_unique == num_data and clean[name].is_monotonic:
            assert not to_numeric and not is_nan
            value = module.add_module(module=EnumerationValue, name=name)

    if value is None:
        assert not to_numeric
        value = module.add_module(module=SamplingValue, name=name)
        is_nan = False
        # print(name, dtype, num_data, num_unique)
        # raise NotImplementedError

    if is_nan:
        value = module.add_module(
            module=NanValue, name=(name + '-nan'), value=value, capacity=module.capacity
        )

    return value
