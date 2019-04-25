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


def identify_value(module, name, dtype, data):
    value = None
    is_nan = False
    num_data = len(data)
    num_unique = data[name].nunique()

    if name in (getattr(module, 'title_label', None), getattr(module, 'gender_label', None), getattr(module, 'name_label', None), getattr(module, 'firstname_label', None), getattr(module, 'lastname_label', None), getattr(module, 'email_label', None)):
        if module.person_value is None:
            value = module.add_module(
                module=PersonValue, name='person', title_label=module.title_label, gender_label=module.gender_label,
                name_label=module.name_label, firstname_label=module.firstname_label,
                lastname_label=module.lastname_label, email_label=module.email_label,
                capacity=module.capacity,
            )
            module.person_value = value
        return value

    elif name in (getattr(module, 'postcode_label', None), getattr(module, 'city_label', None), getattr(module, 'street_label', None)):
        if module.address_value is None:
            value = module.add_module(
                module=AddressValue, name='address', postcode_level=0,
                postcode_label=module.postcode_label, city_label=module.city_label, street_label=module.street_label,
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

    elif dtype.kind == 'M' and num_unique > 2.5 * log(num_data):  # 'm' timedelta
        value = module.add_module(module=DateValue, name=name, capacity=module.capacity)

    elif dtype.kind == 'b' and num_unique > 2.5 * log(num_data):
        value = module.add_module(
            module=CategoricalValue, name=name, categories=[False, True], capacity=module.capacity
        )

    elif dtype.kind == 'O' and num_unique > 2.5 * log(num_data):
        if hasattr(dtype, 'categories'):
            # categorical if dtype has categories
            value = module.add_module(
                module=CategoricalValue, name=name, categories=dtype.categories,
                capacity=module.capacity, pandas_category=True
            )

        else:
            try:
                # datetime if values can be parsed
                is_nan = pd.to_datetime(data[name]).isna().any()
                value = module.add_module(module=DateValue, name=name, capacity=module.capacity)
            except ValueError:
                pass

    if value is None and num_unique > 1:
        # categorical if small number of distinct values
        if num_unique <= 2.5 * log(num_data):
            value = module.add_module(module=CategoricalValue, name=name, capacity=module.capacity)

    if dtype.kind == 'O' and num_unique > sqrt(num_data):
        # numerical if values can be parsed
        numeric_data = pd.to_numeric(data[name], errors='coerce')
        if numeric_data.isna().sum() / len(numeric_data) < 0.25:
            dtype = numeric_data.dtype
            assert dtype.kind in ('f', 'i')
    elif dtype.kind in ('f', 'i'):
        numeric_data = data[name]

    if value is None and dtype.kind in ('f', 'i') and num_unique > 1:
        is_nan = numeric_data.isna().any()
        value = module.add_module(module=ContinuousValue, name=name)

    if value is not None and is_nan:
        value = module.add_module(module=NanValue, name=name, value=value, capacity=module.capacity)

    if value is None:
        if num_unique <= sqrt(num_data) and num_unique > 1:
            # categorical similarity if not too many distinct values
            value = module.add_module(
                module=CategoricalValue, name=name, capacity=module.capacity, similarity_based=True
            )

        elif dtype.kind != 'f' and num_unique == num_data and data[name].is_monotonic:
            # enumeration if it looks like an index
            value = module.add_module(module=EnumerationValue, name=name)

        else:
            # otherwise sample values
            value = module.add_module(module=SamplingValue, name=name)
            # print(name, dtype, num_data, num_unique)
            # raise NotImplementedError

    return value
