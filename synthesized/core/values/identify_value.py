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


CATEGORICAL_THRESHOLD_LOG_MULTIPLIER = 2.5
PARSING_NAN_FRACTION_THRESHOLD = 0.25


def identify_value(module, name, dtype, data):
    value = None

    # ========== Pre-configured values ==========

    # Person value
    if name == getattr(module, 'title_label', None) or \
            name == getattr(module, 'gender_label', None) or \
            name == getattr(module, 'name_label', None) or \
            name == getattr(module, 'firstname_label', None) or \
            name == getattr(module, 'lastname_label', None) or \
            name == getattr(module, 'email_label', None):
        if module.person_value is None:
            value = module.add_module(
                module=PersonValue, name='person', title_label=module.title_label, gender_label=module.gender_label,
                name_label=module.name_label, firstname_label=module.firstname_label,
                lastname_label=module.lastname_label, email_label=module.email_label,
                capacity=module.capacity,
            )
            module.person_value = value

    # Address value
    elif name == getattr(module, 'postcode_label', None) or \
            name == getattr(module, 'city_label', None) or \
            name == getattr(module, 'street_label', None):
        if module.address_value is None:
            value = module.add_module(
                module=AddressValue, name='address', postcode_level=0,
                postcode_label=module.postcode_label, city_label=module.city_label, street_label=module.street_label,
                capacity=module.capacity
            )
            module.address_value = value

    # Compound address value
    elif name == getattr(module, 'address_label', None):
        value = module.add_module(
            module=CompoundAddressValue, name='address', postcode_level=1,
            address_label=module.address_label,
            postcode_regex=module.postcode_regex,
            capacity=module.capacity
        )
        module.address_value = value

    # Identifier value
    elif name == getattr(module, 'identifier_label', None):
        value = module.add_module(module=IdentifierValue, name=name, capacity=module.capacity)
        module.identifier_value = value

    # Return pre-configured value
    if value is not None:
        return value

    # ========== Non-numeric values ==========

    num_data = len(data)
    num_unique = data[name].nunique()
    is_nan = False

    # Categorical value if small number of distinct values
    if num_unique <= CATEGORICAL_THRESHOLD_LOG_MULTIPLIER * log(num_data):
        # is_nan = data[name].isna().any()
        value = module.add_module(module=CategoricalValue, name=name, capacity=module.capacity)

    # Date value
    elif dtype.kind == 'M':  # 'm' timedelta
        is_nan = data[name].isna().any()
        value = module.add_module(module=DateValue, name=name, capacity=module.capacity)

    # Boolean value
    elif dtype.kind == 'b':
        # is_nan = data[name].isna().any()
        value = module.add_module(
            module=CategoricalValue, name=name, categories=[False, True], capacity=module.capacity
        )

    # Continuous value if integer (reduced variability makes similarity-categorical fallback more likely)
    elif dtype.kind == 'i':
        value = module.add_module(module=ContinuousValue, name=name, integer=True)

    # Categorical value if object type has attribute 'categories'
    elif dtype.kind == 'O' and hasattr(dtype, 'categories'):
        # is_nan = data[name].isna().any()
        value = module.add_module(
            module=CategoricalValue, name=name, categories=dtype.categories,
            capacity=module.capacity, pandas_category=True
        )

    # Date value if object type can be parsed
    elif dtype.kind == 'O':
        try:
            date_data = pd.to_datetime(data[name])
            num_nan = date_data.isna().sum()
            if num_nan / num_data < PARSING_NAN_FRACTION_THRESHOLD:
                assert date_data.dtype.kind == 'M'
                value = module.add_module(module=DateValue, name=name, capacity=module.capacity)
                is_nan = num_nan > 0
        except ValueError:
            pass

    # Similarity-based categorical value if not too many distinct values
    elif num_unique <= sqrt(num_data):
        value = module.add_module(
            module=CategoricalValue, name=name, capacity=module.capacity, similarity_based=True
        )

    # Return non-numeric value and handle NaNs if necessary
    if value is not None:
        if is_nan:
            value = module.add_module(
                module=NanValue, name=name, value=value, capacity=module.capacity
            )
        return value

    # ========== Numeric value ==========

    # Try parsing if object type
    if dtype.kind == 'O':
        numeric_data = pd.to_numeric(data[name], errors='coerce')
        num_nan = numeric_data.isna().sum()
        if num_nan / num_data < PARSING_NAN_FRACTION_THRESHOLD:
            assert numeric_data.dtype.kind in ('f', 'i')
            dtype = numeric_data.dtype
            is_nan = num_nan > 0
    elif dtype.kind in ('f', 'i'):
        is_nan = data[name].isna().any()

    # Return numeric value and handle NaNs if necessary
    if dtype.kind in ('f', 'i'):
        value = module.add_module(module=ContinuousValue, name=name)
        if is_nan:
            value = module.add_module(
                module=NanValue, name=name, value=value, capacity=module.capacity
            )
        return value

    # ========== Fallback values ==========

    # Enumeration value if strictly increasing
    if dtype.kind != 'f' and num_unique == num_data and data[name].is_monotonic_increasing:
        value = module.add_module(module=EnumerationValue, name=name)

    # Sampling value otherwise
    else:
        value = module.add_module(module=SamplingValue, name=name)

    return value
