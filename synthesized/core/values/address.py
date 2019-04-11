import numpy as np
import pandas as pd
import re

from .value import Value
from .categorical import CategoricalValue
from ..module import tensorflow_name_scoped


# TODO: postcodes needs one-time scan to collect
# street etc optional


class AddressValue(Value):

    postcode_regex = re.compile(r'^[A-Za-z]{1,2}[0-9]+[A-Za-z]? [0-9]+[A-Za-z]{2}$')

    def __init__(self, name, postcode_level=0, postcode_label=None, capacity=None, city_label=None, street_label=None, postcodes=None):
        super().__init__(name=name)

        if postcode_level < 0 or postcode_level > 2:
            raise NotImplementedError

        self.postcode_level = postcode_level
        self.postcode_label = postcode_label
        self.city_label = city_label
        self.street_label = street_label

        if postcodes is None:
            self.postcodes = None
        else:
            self.postcodes = sorted(postcodes)

        self.streets = {}
        self.cities = {}

        if postcode_label is None:
            self.postcode = None
        else:
            self.postcode = self.add_module(
                module=CategoricalValue, name=postcode_label, categories=self.postcodes,
                capacity=capacity
            )

    def input_size(self):
        if self.postcode is None:
            return 0
        else:
            return self.postcode.input_size()

    def output_size(self):
        if self.postcode is None:
            return 0
        else:
            return self.postcode.output_size()

    def input_labels(self):
        if self.postcode is not None:
            yield from self.postcode.input_labels()
        # if self.street_label is not None:
        #     yield self.street_label

    def output_labels(self):
        if self.postcode is not None:
            yield from self.postcode.output_labels()

    def placeholders(self):
        if self.postcode is not None:
            yield from self.postcode.placeholders()

    def extract(self, data):
        if self.postcodes is None:
            self.postcodes = dict()
            fixed = False
        else:
            self.postcodes = {postcode: list() for postcode in self.postcodes}
            fixed = True

        keys = []
        for n, row in data.iterrows():
            postcode = row[self.postcode_label]
            if not self.__class__.postcode_regex.match(postcode):
                raise ValueError(postcode)
            if self.postcode_level == 0:  # 1-2 letters
                index = 2 - postcode[1].isdigit()
            elif self.postcode_level == 1:
                index = postcode.index(' ')
            elif self.postcode_level == 2:
                index = postcode.index(' ') + 2
            postcode_key = postcode[:index]
            postcode_value = postcode[index:]
            if postcode_key not in self.postcodes:
                if fixed:
                    raise NotImplementedError
                else:
                    self.postcodes[postcode_key] = []
            self.postcodes[postcode_key].append(postcode_value)

            if self.street_label:
                self.streets[postcode_key] = row[self.street_label]

            if self.city_label:
                self.cities[postcode_key] = row[self.city_label]

            keys.append(postcode_key)

        # convert list to ndarray for better performance
        for key, postcode in self.postcodes.items():
            self.postcodes = np.array(self.postcodes[key])

        if self.postcode is not None:
            postcode_data = pd.DataFrame({self.postcode_label: keys})
            self.postcode.extract(data=postcode_data)

    def preprocess(self, data):
        postcodes = []
        for n, row in data.iterrows():
            postcode = row[self.postcode_label]
            if not self.__class__.postcode_regex.match(postcode):
                raise ValueError(postcode)
            if self.postcode_level == 0:  # 1-2 letters
                index = 2 - postcode[1].isdigit()
            elif self.postcode_level == 1:
                index = postcode.index(' ')
            elif self.postcode_level == 2:
                index = postcode.index(' ') + 2
            postcode_key = postcode[:index]
            postcodes.append(postcode_key)

        data[self.postcode_label] = postcodes

        if self.postcode is not None:
            data = self.postcode.preprocess(data=data)
        return data

    def postprocess(self, data):
        if self.postcodes is None or isinstance(self.postcodes, set):
            raise NotImplementedError
        if self.postcode is None:
            postcode = pd.Series(data=np.random.choice(a=list(self.postcodes), size=len(data)), name=self.postcode_label)
        else:
            data = self.postcode.postprocess(data=data)
            postcode = data[self.postcode_label].astype(dtype='str')

        def expand_postcode(key):
            return key + np.random.choice(self.postcodes[key])

        def lookup_city(key):
            return self.cities[key]

        def lookup_street(key):
            return self.streets[key]

        data[self.postcode_label] = postcode.apply(expand_postcode)

        if self.city_label:
            data[self.city_label] = postcode.apply(lookup_city)

        if self.street_label:
            data[self.street_label] = postcode.apply(lookup_street)

        return data

    def features(self, x=None):
        features = super().features(x=x)
        if self.postcode is not None:
            features.update(self.postcode.features(x=x))
        return features

    @tensorflow_name_scoped
    def input_tensor(self, feed=None):
        if self.postcode is None:
            return super().input_tensor(feed=feed)
        else:
            return self.postcode.input_tensor(feed=feed)

    @tensorflow_name_scoped
    def output_tensors(self, x):
        if self.postcode is None:
            return super().output_tensors(x=x)
        else:
            return self.postcode.output_tensors(x=x)

    @tensorflow_name_scoped
    def loss(self, x, feed=None):
        if self.postcode is None:
            return super().loss(x=x, feed=feed)
        else:
            return self.postcode.loss(x=x, feed=feed)
