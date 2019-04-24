import numpy as np
import tensorflow as tf

from synthesized.core import Module
from synthesized.core.values import CategoricalValue, ContinuousValue


def _test_value(value, x, y=None):
    tf.reset_default_graph()
    Module.placeholders = dict()
    value.initialize()
    feed_dict = {value.placeholder: x}
    input_tensor_output = value.input_tensor()
    if y is None:
        output_tensors_output = value.output_tensors(x=input_tensor_output)
        loss_output = value.loss(x=input_tensor_output)
    else:
        output_input = tf.placeholder(dtype=tf.float32, shape=(None, value.output_size()))
        feed_dict[output_input] = y
        output_tensors_output = value.output_tensors(x=output_input)
        loss_output = value.loss(x=output_input)
    initialize = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(fetches=initialize)
        input_tensor = session.run(fetches=input_tensor_output, feed_dict=feed_dict)
        assert input_tensor.shape == (4, value.input_size())
        output_tensors = session.run(fetches=output_tensors_output, feed_dict=feed_dict)
        assert len(output_tensors) == 1
        output_tensor = next(iter(output_tensors.values()))
        assert output_tensor.shape == x.shape == (4,)
        loss = session.run(fetches=loss_output, feed_dict=feed_dict)
        assert loss.shape == ()


def test_categorical():
    value = CategoricalValue(
        name='categorical', categories=list(range(8)), capacity=64, embedding_size=None,
        pandas_category=False, similarity_based=False, weight_decay=0.0, temperature=1.0,
        smoothing=0.1, moving_average=True, similarity_regularization=0.1,
        entropy_regularization=0.1
    )
    _test_value(
        value=value, x=np.random.randint(low=0, high=8, size=(4,)),
        y=np.random.randn(4, value.output_size())
    )


def test_categorical_similarity():
    value = CategoricalValue(
        name='categorical', categories=list(range(8)), capacity=64, embedding_size=None,
        pandas_category=False, similarity_based=True, weight_decay=0.0, temperature=1.0,
        smoothing=0.1, moving_average=True, similarity_regularization=0.1,
        entropy_regularization=0.1
    )
    _test_value(value=value, x=np.random.randint(low=0, high=8, size=(4,)))


def test_continuous():
    value = ContinuousValue(name='continuous', integer=None)
    _test_value(value=value, x=np.random.randn(4,))
