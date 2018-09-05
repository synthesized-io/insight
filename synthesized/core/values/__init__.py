from .address import AddressValue
from .categorical import CategoricalValue
from .continuous import ContinuousValue
from .date import DateValue
from .identifier import IdentifierValue
from .person import PersonValue
from .value import Value


value_modules = dict(
    address=AddressValue,
    categorical=CategoricalValue,
    continuous=ContinuousValue,
    date=DateValue,
    identifier=IdentifierValue,
    person=PersonValue
)


def get_value(self, name, dtype):

    if name == 'name':
        value = self.add_module(
            module='person', modules=value_modules, name='person',
            gender_embedding_size=self.embedding_size, gender_label='gender', name_label='name',
            firstname_label='firstname', lastname_label='lastname', email_label='email'
        )
    elif name in ('gender', 'firstname', 'lastname', 'email'):
        value = None

    elif name == 'postcode':
        value = self.add_module(
            module='address', modules=value_modules, name='address', postcode_level=1,
            postcode_label='postcode', postcode_embedding_size=self.embedding_size,
            street_label='street'
        )
    elif name in ('street',):
        value = None

    elif name == 'account_id':
        value = self.add_module(
            module='identifier', modules=value_modules, name=name,
            num_identifiers=4500, embedding_size=self.id_embedding_size
        )
        self.identifier_value = value

    elif dtype.kind == 'f':
        # float defaults to continuous (positive?)
        value = self.add_module(
            module='continuous', modules=value_modules, name=name
        )

    elif dtype.kind == 'O' and hasattr(dtype, 'categories'):
        # non-float defaults to categorical (requires dtype.categories?)
        value = self.add_module(
            module='categorical', modules=value_modules, name=name,
            categories=dtype.categories, embedding_size=self.embedding_size
        )

    else:
        raise NotImplementedError

    return value
