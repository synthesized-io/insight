import names
import numpy as np
import pandas as pd
import re

from .value import Value
from .categorical import CategoricalValue


# TODO: postcodes needs one-time scan to collect
# street etc optional


class AddressValue(Value):

    postcode_regex = re.compile(r'^[A-Z]{1,2}[0-9][A-Z0-9]? [0-9][A-Z]{2}$')

    def __init__(self, name, postcode_level=0, postcode_label=None, postcodes=None, postcode_embedding_size=None, street_label=None):
        super().__init__(name=name)

        if postcode_level < 0 or postcode_level > 2:
            raise NotImplementedError

        self.postcode_level = postcode_level
        self.postcode_label = postcode_label
        self.street_label = street_label

        if postcode_label is None:
            self.postcodes = None
            self.postcode = None
        else:
            self.postcodes = set(postcodes)
            self.postcode = self.add_module(
                module=CategoricalValue, name=postcode_label, categories=self.postcodes,
                embedding_size=postcode_embedding_size
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

    def labels(self):
        if self.postcode is None:
            return
        else:
            yield from self.postcode.labels()

    def placeholders(self):
        if self.postcode is None:
            return
        else:
            yield from self.postcode.placeholders()

    def preprocess(self, data):
        if self.postcodes is None:
            self.postcodes = dict()
            fixed = False
        elif isinstance(self.postcodes, set):
            self.postcodes = {postcode: list() for postcode in self.postcodes}
            fixed = True
        for n, row in data.iterrows():
            postcode = row[self.postcode_label]
            if not self.__class__.postcode_regex.match(postcode):
                raise NotImplementedError
            if self.postcode_level == 0:  # 1-2 letters
                index = 2 - postcode[1].isdigit()
            elif self.postcode_level == 1:
                index = postcode.index(' ')
            elif self.postcode_level == 2:
                index = postcode.index(' ') + 2
            postcode_key = postcode[:index]
            postcode_value = postcode[index:]
            if fixed and postcode_key not in self.postcodes:
                raise NotImplementedError
            street = row[self.street_label]
            self.postcodes[postcode_key].append((postcode_value, street))
            # data[self.postcode_label][n] = postcode_key
            data.loc[n, self.postcode_label] = postcode_key

        if self.postcode is not None:
            data = self.postcode.preprocess(data=data)
        if self.street_label is not None:
            data = data.drop(labels=self.street_label, axis=1)
        return data

    def postprocess(self, data):
        if self.postcodes is None or isinstance(self.postcodes, set):
            raise NotImplementedError
        if self.postcode is None:
            postcode = np.random.choice(a=list(self.postcodes), size=len(data))
        else:
            data = self.postcode.postprocess(data=data)
            postcode = data[self.postcode_label].astype(dtype='str')
        street = postcode.copy()
        for postcode_key, values in self.postcodes.items():
            postcode_values, streets = list(map(list, zip(*values)))
            mask = (postcode == postcode_key)
            postcode[mask] += np.random.choice(a=postcode_values, size=mask.sum())
            street[mask] = np.random.choice(a=streets, size=mask.sum())
        data[self.postcode_label] = postcode
        data[self.street_label] = street
        return data

    def feature(self, x=None):
        if self.postcode is None:
            return None
        else:
            return self.postcode.feature(x=x)

    def tf_input_tensor(self, feed=None):
        if self.postcode is None:
            return super().tf_input_tensor(feed=feed)
        else:
            return self.postcode.input_tensor(feed=feed)

    def tf_output_tensors(self, x):
        if self.postcode is None:
            return super().tf_output_tensors(x=x)
        else:
            return self.postcode.output_tensors(x=x)

    def tf_loss(self, x, feed=None):
        if self.postcode is None:
            return super().tf_loss(x=x, feed=feed)
        else:
            return self.postcode.loss(x=x, feed=feed)
