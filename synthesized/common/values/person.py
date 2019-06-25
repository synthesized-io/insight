import faker
import numpy as np
import pandas as pd

from .categorical import CategoricalValue
from .value import Value
from ..module import tensorflow_name_scoped


class PersonValue(Value):

    def __init__(self, name, title_label=None, gender_label=None, name_label=None, firstname_label=None,
                 lastname_label=None, email_label=None, capacity=None, dict_cache_size=10000):
        super().__init__(name=name)

        self.title_label = title_label
        self.gender_label = gender_label
        self.name_label = name_label
        self.firstname_label = firstname_label
        self.lastname_label = lastname_label
        self.email_label = email_label
        # Assume the gender are always encoded like M or F or U(???)
        self.title_mapping = {'M': 'Mr', 'F': 'Mrs', 'U': 'female'}

        fkr = faker.Faker(locale='en_GB')
        self.male_first_name_cache = np.array(list({fkr.first_name_male() for _ in range(dict_cache_size)}))
        self.female_first_name_cache = np.array(list({fkr.first_name_female() for _ in range(dict_cache_size)}))
        self.last_name_cache = np.array(list({fkr.last_name() for _ in range(dict_cache_size)}))

        if gender_label is None:
            self.gender = None
        else:
            self.gender = self.add_module(
                module=CategoricalValue, name=gender_label,
                capacity=capacity,
            )

    def input_tensor_size(self):
        if self.gender is None:
            return 0
        else:
            return self.gender.input_tensor_size()

    def output_tensor_size(self):
        if self.gender is None:
            return 0
        else:
            return self.gender.output_tensor_size()

    def input_tensor_labels(self):
        if self.gender is not None:
            yield from self.gender.input_tensor_labels()
        # if self.name_label is not None:
        #     yield self.name_label
        # if self.firstname_label is not None:
        #     yield self.firstname_label
        # if self.lastname_label is not None:
        #     yield self.lastname_label
        # if self.email_label is not None:
        #     yield self.email_label

    def output_tensor_labels(self):
        if self.gender is not None:
            yield from self.gender.output_tensor_labels()

    def placeholders(self):
        if self.gender is not None:
            yield from self.gender.placeholders()

    def extract(self, data):
        if self.gender is not None:
            self.gender.extract(data=data)

    def preprocess(self, data):
        if self.gender is not None:
            data = self.gender.preprocess(data=data)
        return data

    def postprocess(self, data):
        if self.gender is None:
            gender = np.random.choice(a=['female', 'male'], size=len(data))
        else:
            data = self.gender.postprocess(data=data)
            gender = data[self.gender_label]

        def get_first_name(g):
            if g == 'M':
                return np.random.choice(self.male_first_name_cache)
            else:
                return np.random.choice(self.female_first_name_cache)

        title = gender.astype(dtype=str).apply(func=self.title_mapping.__getitem__)
        firstname = gender.astype(dtype=str).apply(func=get_first_name)
        lastname = pd.Series(data=np.random.choice(self.last_name_cache, size=len(data)))

        if self.title_label is not None:
            data.loc[:, self.title_label] = title
        if self.name_label is not None:
            data.loc[:, self.name_label] = firstname.str.cat(others=lastname, sep=' ')
        if self.firstname_label is not None:
            data.loc[:, self.firstname_label] = firstname
        if self.lastname_label is not None:
            data.loc[:, self.lastname_label] = lastname
        if self.email_label is not None:
            # https://email-verify.my-addr.com/list-of-most-popular-email-domains.php
            # we don't want clashes with real emails
            # domain = np.random.choice(a=['gmail.com', 'yahoo.com', 'hotmail.com'], size=len(data))
            data.loc[:, self.email_label] = firstname.str.lower() \
                .str.cat(others=lastname.str.lower(), sep='.')
            data.loc[:, self.email_label] += '@example.com'
        return data

    def features(self, x=None):
        features = super().features(x=x)
        if self.gender is not None:
            features.update(self.gender.features(x=x))
        return features

    @tensorflow_name_scoped
    def input_tensors(self, feed=None):
        if self.gender is None:
            return super().input_tensors(feed=feed)
        else:
            return self.gender.input_tensors(feed=feed)

    @tensorflow_name_scoped
    def output_tensors(self, x):
        if self.gender is None:
            return super().output_tensors(x=x)
        else:
            return self.gender.output_tensors(x=x)

    @tensorflow_name_scoped
    def loss(self, x, feed=None):
        if self.gender is None:
            return super().loss(x=x, feed=feed)
        else:
            return self.gender.loss(x=x, feed=feed)
