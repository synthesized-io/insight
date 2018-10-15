import base64
import os
from datetime import datetime

import tensorflow as tf
from sklearn.base import TransformerMixin

from .module import Module


def _check_license():
    try:
        key_env = 'SYNTHESIZED_KEY'
        key_path = '~/.synthesized/key'
        key_path = os.path.expanduser(key_path)
        print("Copyright (C) Synthesized Ltd. - All Rights Reserved")
        license_key = os.environ.get(key_env, None)
        if license_key is None and os.path.isfile(key_path):
            with open(key_path, 'r') as f:
                license_key = f.readlines()[0].rstrip()
        if license_key is None:
            print('No license key detected (env variable {} or {})'.format(key_env, key_path))
            return False
        else:
            print('License key: ' + license_key)
        license_key_bytes = base64.b16decode(license_key.replace('-', ''))
        key = 13
        n = 247
        plain = ''.join([chr((char ** key) % n) for char in list(license_key_bytes)])
        date = datetime.strptime(plain.split(' ')[1], "%Y-%m-%d")
        now = datetime.now()
        if now < date:
            print('Expires at: ' + str(date))
            return True
        if now >= date:
            print('License has been expired')
            return False
        else:
            print('Expires at: ' + str(date))
            return True
    except Exception as e:
        print(e)
        return False


if not _check_license():
    raise Exception('Failed to load license key')


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

    def tfrecords(self, data, path):
        data = data.copy()
        for value in self.values:
            data = value.preprocess(data=data)
        data = [
            data[label].get_values() for value in self.values for label in value.trainable_labels()
        ]
        options = tf.python_io.TFRecordOptions(
            compression_type=tf.python_io.TFRecordCompressionType.GZIP
        )
        with tf.python_io.TFRecordWriter(path=(path + '.tfrecords'), options=options) as writer:
            for n in range(len(data[0])):
                features = dict()
                # feature_lists = dict()
                i = 0
                for value in self.values:
                    for label in value.trainable_labels():
                        features[label] = value.feature(x=data[i][n])
                        i += 1
                # record = tf.train.SequenceExample(context=tf.train.Features(feature=features), feature_lists=tf.train.FeatureLists(feature_list=feature_lists))
                record = tf.train.Example(features=tf.train.Features(feature=features))
                serialized_record = record.SerializeToString()
                writer.write(record=serialized_record)
