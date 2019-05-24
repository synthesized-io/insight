import numpy as np
import tensorflow as tf
from synthesized.core.encodings import BasicEncoding, GumbelVariationalEncoding, VariationalEncoding


def _test_encoding(encoding):
    assert isinstance(encoding.specification(), dict)
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
        assert encoded.shape[0] == 4 and encoded.shape[1] % 6 == 0
        assert loss.shape == ()
        sampled = session.run(
            fetches=sample_output, feed_dict={sample_input: 4}
        )
        assert sampled.shape[0] == 4 and sampled.shape[1] % 6 == 0


def test_basic():
    encoding = BasicEncoding(name='basic-normal', input_size=8, encoding_size=6, sampling='normal')
    _test_encoding(encoding=encoding)

    encoding = BasicEncoding(
        name='basic-uniform', input_size=8, encoding_size=6, sampling='uniform'
    )
    _test_encoding(encoding=encoding)


def test_gumbel():
    encoding = GumbelVariationalEncoding(
        name='gumbel', input_size=8, encoding_size=6, num_categories=4, temperature=1.0, hard=False,
        beta=1.0
    )
    _test_encoding(encoding=encoding)

    encoding = GumbelVariationalEncoding(
        name='gumbel', input_size=8, encoding_size=6, num_categories=4, temperature=1.0, hard=True,
        beta=1.0
    )
    _test_encoding(encoding=encoding)


def test_variational():
    encoding = VariationalEncoding(name='variational', input_size=8, encoding_size=6, beta=1.0)
    _test_encoding(encoding=encoding)
