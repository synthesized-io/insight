from math import log, sqrt

from .address import AddressValue
from .categorical import CategoricalValue
from .continuous import ContinuousValue
from .date import DateValue
from .enumeration import EnumerationValue
from .gaussian import GaussianValue
from .identifier import IdentifierValue
from .person import PersonValue
from .poisson import PoissonValue
from .sampling import SamplingValue


def identify_value(synthesizer, name, dtype, data):

    if name in (synthesizer.gender_label, synthesizer.name_label, synthesizer.firstname_label, synthesizer.lastname_label, synthesizer.email_label):
        if synthesizer.person_value is None:
            value = synthesizer.add_module(
                module=PersonValue, name='person', gender_label=synthesizer.gender_label,
                gender_embedding_size=synthesizer.embedding_size, name_label=synthesizer.name_label,
                firstname_label=synthesizer.firstname_label, lastname_label=synthesizer.lastname_label,
                email_label=synthesizer.email_label
            )
            synthesizer.person_value = value

    elif name in (synthesizer.postcode_label, synthesizer.street_label):
        if synthesizer.address_value is None:
            value = synthesizer.add_module(
                module=AddressValue, name='address', postcode_level=1,
                postcode_label=synthesizer.postcode_label, postcode_embedding_size=synthesizer.embedding_size,
                street_label=synthesizer.street_label
            )
            synthesizer.address_value = value

    elif name == synthesizer.identifier_label:
        value = synthesizer.add_module(
            module=IdentifierValue, name=name, embedding_size=synthesizer.id_embedding_size
        )
        synthesizer.identifier_value = value

    elif dtype.kind == 'M':  # 'm' timedelta
        if synthesizer.date_value is not None:
            raise NotImplementedError
        value = synthesizer.add_module(module=DateValue, name=name, embedding_size=synthesizer.embedding_size)
        synthesizer.date_value = value

    elif dtype.kind == 'b':
        value = synthesizer.add_module(
            module=CategoricalValue, name=name, categories=[False, True],
            embedding_size=synthesizer.embedding_size
        )

    elif dtype.kind == 'O' and hasattr(dtype, 'categories'):
        value = synthesizer.add_module(
            module=CategoricalValue, name=name, categories=dtype.categories,
            embedding_size=synthesizer.embedding_size
        )

    else:
        num_data = len(data)
        num_unique = data[name].nunique()

        if num_unique <= log(num_data):
            value = synthesizer.add_module(
                module=CategoricalValue, name=name, embedding_size=synthesizer.embedding_size
            )

        elif num_unique <= sqrt(num_data):
            value = synthesizer.add_module(
                module=CategoricalValue, name=name, embedding_size=synthesizer.embedding_size,
                similarity_based=True
            )

        elif dtype.kind != 'f' and num_unique == num_data and data[name].is_monotonic:
            value = synthesizer.add_module(module=EnumerationValue, name=name)

        elif dtype.kind == 'f' or dtype.kind == 'i':
            value = synthesizer.add_module(
                module=ContinuousValue, name=name, integer=(dtype.kind == 'i')
            )

        else:
            print(name, dtype, num_data, num_unique)
            value = synthesizer.add_module(module=SamplingValue, name=name)

    return value
