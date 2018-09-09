from math import log, sqrt

from .address import AddressValue
from .categorical import CategoricalValue
from .continuous import ContinuousValue
from .date import DateValue
from .identifier import IdentifierValue
from .person import PersonValue
from .sampling import SamplingValue
from .value import Value


value_modules = dict(
    address=AddressValue,
    categorical=CategoricalValue,
    continuous=ContinuousValue,
    date=DateValue,
    identifier=IdentifierValue,
    person=PersonValue,
    sampling=SamplingValue
)


def get_value(self, name, dtype, data):

    if name in (self.gender_label, self.name_label, self.firstname_label, self.lastname_label, self.email_label):
        if self.person_value is None:
            value = self.add_module(
                module=PersonValue, name='person', gender_label=self.gender_label,
                gender_embedding_size=self.embedding_size, name_label=self.name_label,
                firstname_label=self.firstname_label, lastname_label=self.lastname_label,
                email_label=self.email_label
            )
            self.person_value = value

    elif name in (self.postcode_label, self.street_label):
        if self.address_value is None:
            value = self.add_module(
                module=AddressValue, name='address', postcode_level=1,
                postcode_label=self.postcode_label, postcode_embedding_size=self.embedding_size,
                street_label=self.street_label
            )
            self.address_value = value

    elif name == self.identifier_label:
        value = self.add_module(
            module=IdentifierValue, name=name, embedding_size=self.id_embedding_size
        )
        self.identifier_value = value

    elif dtype.kind == 'O' and hasattr(dtype, 'categories'):
        # non-float defaults to categorical (requires dtype.categories?)
        value = self.add_module(
            module=CategoricalValue, modules=value_modules, name=name, categories=dtype.categories,
            embedding_size=self.embedding_size
        )

    else:
        num_data = len(data)
        num_unique = data[name].nunique()

        if num_unique <= log(num_data):
            value = self.add_module(
                module=CategoricalValue, modules=value_modules, name=name,
                embedding_size=self.embedding_size
            )

        elif num_unique <= sqrt(num_data):
            value = self.add_module(
                module=CategoricalValue, modules=value_modules, name=name,
                embedding_size=self.embedding_size, similarity_based=True
            )

        elif dtype.kind == 'f':
            value = self.add_module(module=ContinuousValue, modules=value_modules, name=name)

        else:
            value = self.add_module(module='sampling', modules=value_modules, name=name)

    return value
