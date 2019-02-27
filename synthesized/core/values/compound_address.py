import numpy as np
import pandas as pd
import re

from faker import Faker

from .value import Value
from .categorical import CategoricalValue


class CompoundAddressValue(Value):
    def __init__(self, name, postcode_level=0, address_label=None, postcode_regex=None, capacity=None):
        super().__init__(name=name)

        if postcode_level < 0 or postcode_level > 2:
            raise NotImplementedError

        self.postcode_level = postcode_level
        self.address_label = address_label
        self.postcode_regex = postcode_regex

        self.postcodes = None
        self.postcode = self.add_module(
            module=CategoricalValue, name=address_label, categories=self.postcodes,
            capacity=capacity
        )
        self.faker = Faker(locale='en_GB')

    def input_size(self):
        return self.postcode.input_size()

    def output_size(self):
        return self.postcode.output_size()

    def input_labels(self):
        yield from self.postcode.input_labels()

    def output_labels(self):
        yield from self.postcode.output_labels()

    def placeholders(self):
        yield from self.postcode.placeholders()

    def extract(self, data):
        self.postcodes = {}

        postcode_data = pd.DataFrame({self.address_label: data[self.address_label]})
        for n, row in data.iterrows():
            postcode = row[self.address_label]
            m = re.match(self.postcode_regex, postcode)
            if not m:
                postcode_data.loc[n, self.address_label] = None
                continue
            postcode = m.group(1)
            if self.postcode_level == 0:  # 1-2 letters
                index = 2 - postcode[1].isdigit()
            elif self.postcode_level == 1:
                index = postcode.index(' ')
            elif self.postcode_level == 2:
                index = postcode.index(' ') + 2
            else:
                raise ValueError(self.postcode_level)
            postcode_key = postcode[:index]
            postcode_value = postcode[index:]
            if postcode_key not in self.postcodes:
                self.postcodes[postcode_key] = list()
            self.postcodes[postcode_key].append(postcode_value)

            postcode_data.loc[n, self.address_label] = postcode_key

        postcode_data.dropna(inplace=True)
        if self.postcode is not None:
            self.postcode.extract(data=postcode_data)

    def preprocess(self, data):
        for n, row in data.iterrows():
            postcode = row[self.address_label]
            m = re.match(self.postcode_regex, postcode)
            if not m:
                data.loc[n, self.address_label] = None
                continue
            postcode = m.group(1)
            if self.postcode_level == 0:  # 1-2 letters
                index = 2 - postcode[1].isdigit()
            elif self.postcode_level == 1:
                index = postcode.index(' ')
            elif self.postcode_level == 2:
                index = postcode.index(' ') + 2
            else:
                raise ValueError(self.postcode_level)
            postcode_key = postcode[:index]
            data.loc[n, self.address_label] = postcode_key

        data.dropna(inplace=True)
        if self.postcode is not None:
            data = self.postcode.preprocess(data=data)
        return data

    def postprocess(self, data):
        if self.postcodes is None or isinstance(self.postcodes, set):
            raise NotImplementedError
        data = self.postcode.postprocess(data=data)
        postcode = data[self.address_label].astype(dtype='str')
        for postcode_key, postcode_values in self.postcodes.items():
            mask = (postcode == postcode_key)
            postcode[mask] += np.random.choice(a=postcode_values, size=mask.sum())

        def expand_address(p):
            address_parts = self.faker.address().split('\n')
            address_parts = address_parts[:-1]
            address_parts.append(p)
            return ', '.join(address_parts)

        data[self.address_label] = postcode.apply(expand_address)
        return data

    def feature(self, x=None):
        return self.postcode.feature(x=x)

    def tf_input_tensor(self, feed=None):
        return self.postcode.input_tensor(feed=feed)

    def tf_output_tensors(self, x):
        return self.postcode.output_tensors(x=x)

    def tf_loss(self, x, feed=None):
        return self.postcode.loss(x=x, feed=feed)
