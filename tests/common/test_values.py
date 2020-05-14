import numpy as np
import tensorflow as tf
import pandas as pd

from synthesized.common import Value
from synthesized.common.values import CategoricalValue, ContinuousValue


def _test_value(value: Value, x: np.ndarray, y: np.ndarray = None):
    assert isinstance(value.specification(), dict)
    with tf.summary.record_if(False):
        value.build()
        input_tensor_output = [tf.constant(value=x)]
        unified_tensor_output = value.unify_inputs(xs=input_tensor_output)
        if y is None:
            output_tensors_output = value.output_tensors(y=unified_tensor_output)
            loss_output = value.loss(y=unified_tensor_output, xs=input_tensor_output)
        else:
            output_input = tf.constant(y, dtype=tf.float32, shape=(x.shape[0], value.learned_output_size()))
            output_tensors_output = value.output_tensors(y=output_input)
            loss_output = value.loss(y=output_input, xs=input_tensor_output)

        input_tensor = input_tensor_output[0]
        assert input_tensor.shape == (4,)
        output_tensor = output_tensors_output[0]
        assert output_tensor.shape == x.shape == (4,)
        loss = loss_output
        assert loss.shape == ()

        variables = value.get_variables()
        value.set_variables(variables)


def test_categorical():
    value = CategoricalValue(
        name='categorical', weight=5.0, categories=list(range(8)), probabilities=None, capacity=64,
        embedding_size=None, pandas_category=False, similarity_based=False,
        temperature=1.0, moving_average=False,
    )
    cat_values = np.random.randint(low=0, high=8, size=(4,))
    value.extract(pd.DataFrame({'categorical': cat_values}))
    _test_value(
        value=value, x=cat_values,
        y=np.random.randn(4, value.learned_output_size())
    )


def test_categorical_similarity():
    value = CategoricalValue(
        name='categorical', weight=5.0, categories=list(range(8)), probabilities=None, capacity=64,
        embedding_size=None, pandas_category=False, similarity_based=True,
        temperature=1.0, moving_average=True
    )
    cat_values = np.random.randint(low=0, high=8, size=(4,))
    value.extract(pd.DataFrame({'categorical': cat_values}))
    _test_value(
        value=value, x=cat_values,
        y=np.random.randn(4, value.learned_output_size())
    )


def test_continuous():
    value = ContinuousValue(
        name='continuous', weight=1.0, integer=None,
        positive=None, nonnegative=None
    )
    _test_value(value=value, x=np.random.randn(4,))
