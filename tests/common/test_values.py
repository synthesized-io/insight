import numpy as np
import tensorflow as tf

from synthesized.common import Value
from synthesized.common.values import CategoricalValue, ContinuousValue


def _test_value(value: Value, x: np.ndarray, y: np.ndarray = None):
    assert isinstance(value.specification(), dict)
    tf.reset_default_graph()
    value.initialize()
    feed_dict = {value.placeholder: x}
    input_tensor_output = value.input_tensors()
    unified_tensor_output = value.unify_inputs(xs=input_tensor_output)
    if y is None:
        output_tensors_output = value.output_tensors(y=unified_tensor_output)
        loss_output = value.loss(y=unified_tensor_output, xs=input_tensor_output)
    else:
        output_input = tf.placeholder(dtype=tf.float32, shape=(None, value.learned_output_size()))
        feed_dict[output_input] = y
        output_tensors_output = value.output_tensors(y=output_input)
        loss_output = value.loss(y=output_input, xs=[x])
    initialize = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(fetches=initialize)
        input_tensor = session.run(fetches=input_tensor_output[0], feed_dict=feed_dict)
        assert input_tensor.shape == (4,)
        output_tensors = session.run(fetches=output_tensors_output, feed_dict=feed_dict)
        assert len(output_tensors) == 1
        output_tensor = next(iter(output_tensors))
        assert output_tensor.shape == x.shape == (4,)
        loss = session.run(fetches=loss_output, feed_dict=feed_dict)
        assert loss.shape == ()


def test_categorical():
    value = CategoricalValue(
        name='categorical', weight=5.0, categories=list(range(8)), probabilities=None, capacity=64,
        embedding_size=None, pandas_category=False, similarity_based=False, weight_decay=0.0,
        temperature=1.0, moving_average=False,
    )
    _test_value(
        value=value, x=np.random.randint(low=0, high=8, size=(4,)),
        y=np.random.randn(4, value.learned_output_size())
    )


def test_categorical_similarity():
    value = CategoricalValue(
        name='categorical', weight=5.0, categories=list(range(8)), probabilities=None, capacity=64,
        embedding_size=None, pandas_category=False, similarity_based=True, weight_decay=0.0,
        temperature=1.0, moving_average=True
    )
    _test_value(
        value=value, x=np.random.randint(low=0, high=8, size=(4,)),
        y=np.random.randn(4, value.learned_output_size())
    )


def test_continuous():
    value = ContinuousValue(
        name='continuous', weight=1.0, integer=None,
        positive=None, nonnegative=None
    )
    _test_value(value=value, x=np.random.randn(4,))
