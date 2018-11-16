import numpy as np
import tensorflow as tf
from synthesized.core.encodings import BasicEncoding, VariationalEncoding


def _test_encoding(encoding):
    tf.reset_default_graph()
    encoding.initialize()
    encode_input = tf.placeholder(dtype=tf.float32, shape=(None, 8))
    encode_output = encoding.encode(x=encode_input, encoding_loss=True)
    sample_input = tf.placeholder(dtype=tf.int64, shape=())
    sample_output = encoding.sample(n=sample_input)
    initialize = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(fetches=initialize)
        encoded, loss = session.run(
            fetches=encode_output, feed_dict={encode_input: np.random.randn(4, 8)}
        )
        assert encoded.shape == (4, 6)
        assert loss.shape == ()
        sampled = session.run(
            fetches=sample_output, feed_dict={sample_input: 4}
        )
        assert sampled.shape == (4, 6)


def test_basic_normal():
    encoding = BasicEncoding(name='basic', input_size=8, encoding_size=6, sampling='normal')
    _test_encoding(encoding=encoding)


def test_basic_uniform():
    encoding = BasicEncoding(name='basic', input_size=8, encoding_size=6, sampling='uniform')
    _test_encoding(encoding=encoding)


def test_variational():
    encoding = VariationalEncoding(name='basic', input_size=8, encoding_size=6, beta=1.0)
    _test_encoding(encoding=encoding)
