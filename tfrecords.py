import sys
import pandas as pd
import tensorflow as tf
from synthesized.common import BasicSynthesizer


path = 'data/' + sys.argv[1]

data = pd.read_csv(path + '.csv')
data = data.dropna()

synthesizer = BasicSynthesizer(data=data)

data = synthesizer.preprocess(data=data)
data = [
    data[label].get_values() for value in synthesizer.values
    for label in value.trainable_labels()
]
options = tf.python_io.TFRecordOptions(
    compression_type=tf.python_io.TFRecordCompressionType.GZIP
)
with tf.python_io.TFRecordWriter(path=(path + '.tfrecords'), options=options) as writer:
    for n in range(len(data[0])):
        features = dict()
        # feature_lists = dict()
        i = 0
        row = dict()
        for value in synthesizer.values:
            for label in value.labels():
                row[label] = data[i][n]
                i += 1
        for value in synthesizer.values:
            features.update(value.features(x=row))
        # record = tf.train.SequenceExample(context=tf.train.Features(feature=features), feature_lists=tf.train.FeatureLists(feature_list=feature_lists))
        record = tf.train.Example(features=tf.train.Features(feature=features))
        serialized_record = record.SerializeToString()
        writer.write(record=serialized_record)
