import unittest
import numpy as np
import tensorflow as tf
from synthesized.core.encodings import BasicEncoding, VariationalEncoding


class TestEncodings(unittest.TestCase):

    def _test_encoding(self, encoding):
        tf.reset_default_graph()
        encoding.initialize()
        encode_input = tf.placeholder(dtype=tf.float32, shape=(None, 8))
        encode_output = encoding.encode(x=encode_input, encoding_loss=True)
        sample_input = tf.placeholder(dtype=tf.int64, shape=())
        sample_output = encoding.sample(n=sample_input)
        initialize = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(fetches=initialize)
            encoded = session.run(
                fetches=encode_output, feed_dict={encode_input: np.random.randn(4, 8)}
            )
            assert encoded.shape == (4, 8)
            sampled = session.run(
                fetches=sample_output, feed_dict={sample_input: 4}
            )
            assert sampled.shape == (4, 8)

    def test_basic_normal(self):
        encoding = BasicEncoding(name='basic', encoding_size=8, sampling='normal')
        self._test_encoding(encoding=encoding)

    def test_basic_uniform(self):
        encoding = BasicEncoding(name='basic', encoding_size=8, sampling='uniform')
        self._test_encoding(encoding=encoding)

    def test_variational(self):
        encoding = VariationalEncoding(name='basic', encoding_size=8, loss_weight=1.0)
        self._test_encoding(encoding=encoding)
