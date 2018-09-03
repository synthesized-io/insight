from .basic_synthesizer import BasicSynthesizer
from .values import value_modules


class NameAddressSynthesizer(BasicSynthesizer):

    def get_value(self, name, dtype):
        if name == 'name':
            value = self.add_module(
                module='person', modules=value_modules, name='person',
                gender_embedding_size=self.embedding_size, gender_label='gender', name_label='name',
                firstname_label='firstname', lastname_label='lastname', email_label='email'
            )

        elif name == 'postcode':
            value = self.add_module(
                module='address', modules=value_modules, name='address', postcode_level=1,
                postcode_label='postcode', postcodes=('CR2', 'CR3', 'CS2'),
                postcode_embedding_size=self.embedding_size, street_label='street'
            )

        elif name in ('gender', 'firstname', 'lastname', 'email', 'street'):
            value = None

        else:
            value = super().get_value(name=name, dtype=dtype)

        return value
