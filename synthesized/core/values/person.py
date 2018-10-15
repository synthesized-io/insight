import names
import numpy as np
import pandas as pd

from .value import Value
from .categorical import CategoricalValue


class PersonValue(Value):

    def __init__(self, name, gender_label=None, gender_embedding_size=None, name_label=None, firstname_label=None, lastname_label=None, email_label=None):
        super().__init__(name=name)

        self.gender_label = gender_label
        self.name_label = name_label
        self.firstname_label = firstname_label
        self.lastname_label = lastname_label
        self.email_label = email_label

        if gender_label is None:
            self.gender = None
        else:
            self.gender = self.add_module(
                module=CategoricalValue, name=gender_label, categories=['female', 'male'],
                embedding_size=gender_embedding_size
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

    def labels(self):
        if self.gender is not None:
            yield from self.gender.labels()
        if self.name_label is not None:
            yield self.name_label
        if self.firstname_label is not None:
            yield self.firstname_label
        if self.lastname_label is not None:
            yield self.lastname_label
        if self.email_label is not None:
            yield self.email_label

    def trainable_labels(self):
        if self.gender is None:
            return
        else:
            yield from self.gender.trainable_labels()

    def placeholders(self):
        if self.gender is None:
            return
        else:
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
        firstname = gender.astype(dtype=str).apply(func=names.get_first_name)
        lastname = pd.Series(data=(names.get_last_name() for _ in range(len(data))))
        if self.name_label is not None:
            data[self.name_label] = firstname.str.cat(others=lastname, sep=' ')
        if self.firstname_label is not None:
            data[self.firstname_label] = firstname
        if self.lastname_label is not None:
            data[self.lastname_label] = lastname
        if self.email_label is not None:
            # https://email-verify.my-addr.com/list-of-most-popular-email-domains.php
            domain = np.random.choice(a=['gmail.com', 'yahoo.com', 'hotmail.com'], size=len(data))
            data[self.email_label] = firstname.str.lower() \
                .str.cat(others=lastname.str.lower(), sep='.') \
                .str.cat(others=domain, sep='@')
        return data

    def feature(self, x=None):
        if self.gender is None:
            return None
        else:
            return self.gender.feature(x=x)

    def tf_input_tensor(self, feed=None):
        if self.gender is None:
            return super().tf_input_tensor(feed=feed)
        else:
            return self.gender.input_tensor(feed=feed)

    def tf_output_tensors(self, x):
        if self.gender is None:
            return super().tf_output_tensors(x=x)
        else:
            return self.gender.output_tensors(x=x)

    def tf_loss(self, x, feed=None):
        if self.gender is None:
            return super().tf_loss(x=x, feed=feed)
        else:
            return self.gender.loss(x=x, feed=feed)