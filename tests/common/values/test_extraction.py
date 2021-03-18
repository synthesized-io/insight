import numpy as np

from synthesized.common.values import (AssociatedCategoricalValue, CategoricalValue, ContinuousValue, DateValue,
                                       ValueExtractor)
from synthesized.metadata import DataFrameMeta
from synthesized.metadata.value import AssociatedCategorical, DateTime, Integer
from synthesized.model import DataFrameModel
from synthesized.model.models import AssociatedHistogram, Histogram, KernelDensityEstimate


def extract_value_from_model(model):
    df_meta = DataFrameMeta("df_meta")
    df_model = DataFrameModel(meta=df_meta)
    df_model[model.name] = model
    return ValueExtractor.extract(df_model)


def test_hist_extraction():
    meta = Integer(name="hist", categories=[0, 1, 2, 3], nan_freq=0.1)
    model = Histogram(meta=meta)
    df_value = extract_value_from_model(model)

    assert isinstance(df_value["hist"], CategoricalValue)
    assert df_value["hist"].num_categories == 4
    assert isinstance(df_value["hist_nan"], CategoricalValue)
    assert len(df_value) == 2


def test_kde_extraction():
    meta = Integer(name='kde', nan_freq=0.1)
    model = KernelDensityEstimate(meta=meta)
    df_value = extract_value_from_model(model)

    assert isinstance(df_value["kde"], ContinuousValue)
    assert isinstance(df_value["kde_nan"], CategoricalValue)
    assert len(df_value) == 2


def test_date_kde_extraction():
    meta = DateTime(name='kde', nan_freq=0.1)
    model = KernelDensityEstimate(meta=meta)
    df_value = extract_value_from_model(model)

    assert isinstance(df_value["kde"], DateValue)
    assert isinstance(df_value["kde_nan"], CategoricalValue)
    assert len(df_value) == 2


def test_associated_value_extraction():
    metas = [Integer(name=str(i), categories=[0, 1], nan_freq=0.1) for i in range(2)]
    meta = AssociatedCategorical(name="associated", children=metas, binding_mask=np.ones((2, 2)))
    model = AssociatedHistogram(meta=meta)
    df_value = extract_value_from_model(model)

    assert isinstance(df_value["associated"], AssociatedCategoricalValue)
    assert isinstance(df_value["0_nan"], CategoricalValue)
    assert isinstance(df_value["1_nan"], CategoricalValue)
    assert len(df_value) == 3
