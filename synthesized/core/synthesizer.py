import base64
import os
import sys
from datetime import datetime

import tensorflow as tf
from sklearn.base import TransformerMixin

from .module import Module

print("Copyright (C) Synthesized Ltd. - All Rights Reserved", file=sys.stderr)
license_key = os.environ.get('SYNTHESIZED_KEY', None)
if license_key is None:
    raise Exception('No license key detected (check env variable SYNTHESIZED_KEY)')
else:
    print('License key: ' + license_key)
license_key_bytes = base64.b16decode(license_key.replace('-', ''))
key = 13
n = 247
plain = ''.join([chr((char ** key) % n) for char in list(license_key_bytes)])
date = datetime.strptime(plain.split(' ')[1], "%Y-%m-%d")
now = datetime.now()
if now >= date:
    raise Exception('License has been expired')
else:
    print('Expires at: ' + str(date), file=sys.stderr)


class Synthesizer(Module, TransformerMixin):

    def __init__(self, name):
        super().__init__(name=name)

    def get_values(self):
        raise NotImplementedError

    def learn(self, data=None, filenames=None, verbose=False):
        raise NotImplementedError

    def synthesize(self, n):
        raise NotImplementedError

    def transform(self, X, **transform_params):
        raise NotImplementedError

    def fit(self, X, y=None, **fit_params):
        assert y is None and not fit_params
        self.learn(data=X)
        return self

    def tfrecords(self, data, name):
        data = [data[value.name].get_values() for value in self.get_values()]
        options = tf.python_io.TFRecordOptions(
            compression_type=tf.python_io.TFRecordCompressionType.GZIP
        )
        with tf.python_io.TFRecordWriter(path=(name + '.tfrecords'), options=options) as writer:
            for n in range(len(data[0])):
                features = dict()
                # feature_lists = dict()
                for value, d in zip(self.get_values(), data):
                    features[value.name] = value.feature(x=d[n])
                # record = tf.train.SequenceExample(context=tf.train.Features(feature=features), feature_lists=tf.train.FeatureLists(feature_list=feature_lists))
                record = tf.train.Example(features=tf.train.Features(feature=features))
                serialized_record = record.SerializeToString()
                writer.write(record=serialized_record)
