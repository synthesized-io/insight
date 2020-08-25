import pytest

import numpy as np
import pandas as pd
import tensorflow as tf

from synthesized.common.values import Value, CategoricalValue, ContinuousValue, NanValue, DateValue
from synthesized.metadata import DateMeta


@pytest.mark.unit
def _test_value(value: Value, x: np.ndarray, y: np.ndarray = None):
    assert isinstance(value.specification(), dict)
    n = len(x)
    with tf.summary.record_if(False):
        value.build()
        input_tensor_output = [tf.constant(x)]
        unified_tensor_output = value.unify_inputs(xs=input_tensor_output)
        if y is None:
            output_tensors_output = value.output_tensors(y=unified_tensor_output)
            loss_output = value.loss(y=unified_tensor_output, xs=input_tensor_output)
        else:
            output_input = tf.constant(y, dtype=tf.float32, shape=(x.shape[0], value.learned_output_size()))
            output_tensors_output = value.output_tensors(y=output_input)
            loss_output = value.loss(y=output_input, xs=input_tensor_output)

        input_tensor = input_tensor_output[0]
        assert input_tensor.shape == (n,)
        output_tensor = output_tensors_output[0]
        assert output_tensor.shape == x.shape == (n,)
        loss = loss_output
        assert loss.shape == ()

        variables = value.get_variables()
        value.set_variables(variables)


@pytest.mark.unit
def test_categorical():
    value = CategoricalValue(
        name='categorical', num_categories=8, probabilities=None,
        embedding_size=None, similarity_based=False
    )
    cat_values = np.random.randint(low=0, high=8, size=(4,))
    _test_value(
        value=value, x=cat_values,
        y=np.random.randn(4, value.learned_output_size())
    )


@pytest.mark.unit
def test_categorical_similarity():
    value = CategoricalValue(
        name='categorical', num_categories=8, probabilities=None,
        embedding_size=None, similarity_based=True
    )
    cat_values = np.random.randint(low=0, high=8, size=(4,))
    _test_value(
        value=value, x=cat_values,
        y=np.random.randn(4, value.learned_output_size())
    )


@pytest.mark.unit
def test_continuous():
    value = ContinuousValue(name='continuous')
    _test_value(value=value, x=np.random.randn(4,))


@pytest.mark.unit
def test_nan():
    cont_value = ContinuousValue(name='continuous')
    value = NanValue(name='nan', value=cont_value)

    n = 10
    x = np.random.randn(n)
    _test_value(value=value,
                x=np.where(x > 0, x, np.nan).astype(np.float32),
                y=np.random.randn(n, value.learned_output_size()).astype(np.float32)
                )


@pytest.mark.unit
def test_nan_inf():
    cont_value = ContinuousValue(name='continuous')
    value = NanValue(name='nan', value=cont_value)

    n = 10
    x = np.array([0.2, -0.1, np.NaN, 0.0, 0.8, np.Inf, -0.6, -np.Inf, 0.13, -0.4], dtype=np.float32)
    _test_value(
        value=value,
        x=x,
        y=np.random.randn(n, value.learned_output_size()).astype(np.float32)
    )


@pytest.mark.unit
def test_date():
    value = DateValue(name='date')

    n = 100
    date0 = np.datetime64('2017-01-01 00:00:00')
    df = pd.DataFrame([date0 + np.random.randint(1000, 1_000_000) for _ in range(n)], columns=['date'])

    meta = DateMeta('date')

    meta.extract(df)
    df = meta.preprocess(df)
    input_tensor_output = [df[c] for c in df.columns]
    unified_tensor_output = value.unify_inputs(xs=input_tensor_output)
    assert unified_tensor_output.shape[0] == n

    variables = value.get_variables()
    value.set_variables(variables)


