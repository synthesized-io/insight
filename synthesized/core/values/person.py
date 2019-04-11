import names
import numpy as np
import pandas as pd

from .value import Value
from .categorical import CategoricalValue
from ..module import tensorflow_name_scoped


class PersonValue(Value):

    def __init__(self, name, title_label=None, gender_label=None, name_label=None, firstname_label=None, lastname_label=None, email_label=None, capacity=None):
        super().__init__(name=name)

        self.title_label = title_label
        self.gender_label = gender_label
        self.name_label = name_label
        self.firstname_label = firstname_label
        self.lastname_label = lastname_label
        self.email_label = email_label
        # Assume the gender are always encoded like M or F
        self.gender_mapping = {'M': 'male', 'F': 'female'}
        self.title_mapping = {'M': 'Mr', 'F': 'Mrs'}

        if gender_label is None:
            self.gender = None
        else:
            self.gender = self.add_module(
                module=CategoricalValue, name=gender_label,
                capacity=capacity,
            )

    def input_size(self):
        if self.gender is None:
            return 0
        else:
            return self.gender.input_size()

    def output_size(self):
        if self.gender is None:
            return 0
        else:
            return self.gender.output_size()

    def input_labels(self):
        if self.gender is not None:
            yield from self.gender.input_labels()
        # if self.name_label is not None:
        #     yield self.name_label
        # if self.firstname_label is not None:
        #     yield self.firstname_label
        # if self.lastname_label is not None:
        #     yield self.lastname_label
        # if self.email_label is not None:
        #     yield self.email_label

    def output_labels(self):
        if self.gender is not None:
            yield from self.gender.output_labels()

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
        title = gender.astype(dtype=str).apply(func=lambda g: self.title_mapping[g])
        firstname = gender.astype(dtype=str).apply(func=lambda g: names.get_first_name(self.gender_mapping[g]))
        lastname = pd.Series(data=(names.get_last_name() for _ in range(len(data))))
        if self.title_label is not None:
            data[self.title_label] = title
        if self.name_label is not None:
            data[self.name_label] = firstname.str.cat(others=lastname, sep=' ')
        if self.firstname_label is not None:
            data[self.firstname_label] = firstname
        if self.lastname_label is not None:
            data[self.lastname_label] = lastname
        if self.email_label is not None:
            # https://email-verify.my-addr.com/list-of-most-popular-email-domains.php
            # we don't want clashes with real emails
            # domain = np.random.choice(a=['gmail.com', 'yahoo.com', 'hotmail.com'], size=len(data))
            data[self.email_label] = firstname.str.lower() \
                .str.cat(others=lastname.str.lower(), sep='.')
            data[self.email_label] += '@example.com'
        return data

    def features(self, x=None):
        features = super().features(x=x)
        if self.gender is not None:
            features.update(self.gender.features(x=x))
        return features

    @tensorflow_name_scoped
    def input_tensor(self, feed=None):
        if self.gender is None:
            return super().input_tensor(feed=feed)
        else:
            return self.gender.input_tensor(feed=feed)

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
