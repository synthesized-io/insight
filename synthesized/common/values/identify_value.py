from math import log, sqrt

import pandas as pd


CATEGORICAL_THRESHOLD_LOG_MULTIPLIER = 2.5
PARSING_NAN_FRACTION_THRESHOLD = 0.25


def identify_value(module, name, dtype, data):
    value = None

    categorical_kwargs = dict()
    continuous_kwargs = dict()
    nan_kwargs = dict()
    categorical_kwargs['capacity'] = module.capacity
    nan_kwargs['capacity'] = module.capacity
    categorical_kwargs['weight_decay'] = module.weight_decay
    nan_kwargs['weight_decay'] = module.weight_decay
    categorical_kwargs['weight'] = module.categorical_weight
    nan_kwargs['weight'] = module.categorical_weight
    continuous_kwargs['weight'] = module.continuous_weight
    categorical_kwargs['temperature'] = module.temperature
    categorical_kwargs['smoothing'] = module.smoothing
    categorical_kwargs['moving_average'] = module.moving_average
    categorical_kwargs['similarity_regularization'] = module.similarity_regularization
    categorical_kwargs['entropy_regularization'] = module.entropy_regularization

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
                module='person', name='person', title_label=module.title_label,
                gender_label=module.gender_label,
                name_label=module.name_label, firstname_label=module.firstname_label,
                lastname_label=module.lastname_label, email_label=module.email_label,
                capacity=module.capacity, weight_decay=module.weight_decay
            )
            module.person_value = value

    # Address value
    elif name == getattr(module, 'postcode_label', None) or \
            name == getattr(module, 'city_label', None) or \
            name == getattr(module, 'street_label', None):
        if module.address_value is None:
            value = module.add_module(
                module='address', name='address', postcode_level=0,
                postcode_label=module.postcode_label, city_label=module.city_label,
                street_label=module.street_label,
                capacity=module.capacity, weight_decay=module.weight_decay
            )
            module.address_value = value

    # Compound address value
    elif name == getattr(module, 'address_label', None):
        value = module.add_module(
            module='compound_address', name='address', postcode_level=1,
            address_label=module.address_label, postcode_regex=module.postcode_regex,
            capacity=module.capacity, weight_decay=module.weight_decay
        )
        module.address_value = value

    # Identifier value
    elif name == getattr(module, 'identifier_label', None):
        value = module.add_module(
            module='identifier', name=name, capacity=module.capacity,
            weight_decay=module.weight_decay
        )
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
        value = module.add_module(module='categorical', name=name, **categorical_kwargs)

    # Date value
    elif dtype.kind == 'M':  # 'm' timedelta
        is_nan = data[name].isna().any()
        value = module.add_module(
            module='date', name=name, capacity=module.capacity, weight_decay=module.weight_decay
        )

    # Boolean value
    elif dtype.kind == 'b':
        # is_nan = data[name].isna().any()
        value = module.add_module(
            module='categorical', name=name, categories=[False, True], **categorical_kwargs
        )

    # Continuous value if integer (reduced variability makes similarity-categorical more likely)
    elif dtype.kind == 'i':
        value = module.add_module(module='continuous', name=name, integer=True, **continuous_kwargs)

    # Categorical value if object type has attribute 'categories'
    elif dtype.kind == 'O' and hasattr(dtype, 'categories'):
        # is_nan = data[name].isna().any()
        value = module.add_module(
            module='categorical', name=name, pandas_category=True, categories=dtype.categories,
            **categorical_kwargs
        )

    # Date value if object type can be parsed
    elif dtype.kind == 'O':
        try:
            date_data = pd.to_datetime(data[name])
            num_nan = date_data.isna().sum()
            if num_nan / num_data < PARSING_NAN_FRACTION_THRESHOLD:
                assert date_data.dtype.kind == 'M'
                value = module.add_module(
                    module='date', name=name, capacity=module.capacity,
                    weight_decay=module.weight_decay
                )
                is_nan = num_nan > 0
        except ValueError:
            pass

    # Similarity-based categorical value if not too many distinct values
    elif num_unique <= sqrt(num_data):
        value = module.add_module(
            module='categorical', name=name, similarity_based=True, **categorical_kwargs
        )

    # Return non-numeric value and handle NaNs if necessary
    if value is not None:
        if is_nan:
            value = module.add_module(module='nan', name=name, value=value, **nan_kwargs)
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
        value = module.add_module(module='continuous', name=name, **continuous_kwargs)
        if is_nan:
            value = module.add_module(module='nan', name=name, value=value, **nan_kwargs)
        return value

    # ========== Fallback values ==========

    # Enumeration value if strictly increasing
    if dtype.kind != 'f' and num_unique == num_data and data[name].is_monotonic_increasing:
        value = module.add_module(module='enumeration', name=name)

    # Sampling value otherwise
    else:
        value = module.add_module(module='sampling', name=name)

    return value
