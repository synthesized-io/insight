from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf
import pandas as pd

from synthesized.common.values import Value
from synthesized.common.values import CategoricalValue, ContinuousValue, NanValue, DateValue


def _test_value(value: Value, x: np.ndarray, y: np.ndarray = None):
    assert isinstance(value.specification(), dict)
    n = len(x)
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
        assert input_tensor.shape == (n,)
        output_tensor = output_tensors_output[0]
        assert output_tensor.shape == x.shape == (n,)
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


def test_nan():
    cont_value = ContinuousValue(
        name='continuous', weight=1.0, integer=None,
        positive=None, nonnegative=None)

    value = NanValue(name='nan', value=cont_value, capacity=32, weight=1.0)

    n = 10
    x = np.random.randn(n)
    _test_value(value=value,
                x=np.where(x > 0, x, np.nan).astype(np.float32),
                y=np.random.randn(n, value.learned_output_size()).astype(np.float32)
                )


def test_date():
    categorical_kwargs = dict(weight=1., capacity=32, temperature=1., moving_average=False)
    continuous_kwargs = dict(weight=1.)

    value = DateValue(name='date', categorical_kwargs=categorical_kwargs, continuous_kwargs=continuous_kwargs)

    n = 100
    date0 = np.datetime64('2017-01-01 00:00:00')
    df = pd.DataFrame([date0 + np.random.randint(1000, 1_000_000) for _ in range(n)], columns=['date'])
    value.extract(df)
    df = value.preprocess(df)
    input_tensor_output = [tf.constant(value=df[c], dtype=tf.float32) for c in df.columns]
    unified_tensor_output = value.unify_inputs(xs=input_tensor_output)
    assert unified_tensor_output.shape[0] == n

    variables = value.get_variables()
    value.set_variables(variables)


