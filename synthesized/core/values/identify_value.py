from math import log, sqrt

from .address import AddressValue
from .categorical import CategoricalValue
from .continuous import ContinuousValue
from .date import DateValue
from .enumeration import EnumerationValue
from .identifier import IdentifierValue
from .person import PersonValue
from .probability import ProbabilityValue
from .sampling import SamplingValue


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

        if num_unique <= log(num_data):
            value = module.add_module(module=CategoricalValue, name=name, capacity=module.capacity)

        elif num_unique <= sqrt(num_data):
            value = module.add_module(
                module=CategoricalValue, name=name, capacity=module.capacity, similarity_based=True
            )

        elif dtype.kind == 'f' and (data[name] <= 1.0).all() and (data[name] >= 0.0).all():
            value = module.add_module(module=ProbabilityValue, name=name)

        elif dtype.kind != 'f' and num_unique == num_data and data[name].is_monotonic:
            value = module.add_module(module=EnumerationValue, name=name)

        elif dtype.kind == 'f' or dtype.kind == 'i':
            value = module.add_module(
                module=ContinuousValue, name=name, integer=(dtype.kind == 'i')
            )

        else:
            value = module.add_module(module=SamplingValue, name=name)
            print(name, dtype, num_data, num_unique)
            raise NotImplementedError

    return value
