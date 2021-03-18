import random

import pandas as pd
import pytest

from synthesized.api import Address, Bank, ContinuousModel, DiscreteModel, HighDimSynthesizer, MetaExtractor, Person
from synthesized.config import AddressLabels, BankLabels, PersonLabels
from synthesized.metadata.value import Address as _Address
from synthesized.metadata.value import Bank as _Bank
from synthesized.metadata.value import Person as _Person
from synthesized.model import ContinuousModel as _ContinuousModel
from synthesized.model import DiscreteModel as _DiscreteModel


@pytest.fixture()
def df():
    df = pd.DataFrame({
        "a": random.choices([0, 1, 2], k=100),
        "b": random.choices(["1", "2", "q"], k=100),
        "c": random.choices(["blah", "bluh", "blih"], k=100)
    })

    return df

@pytest.fixture()
def df_meta(df):
    return MetaExtractor.extract(df)


def test_extraction(df_meta):
    assert hasattr(df_meta, "_df_meta")


def test_annotations(df):
    annotations = [Bank(BankLabels(bic_label="a")), Address(AddressLabels(house_number_label="b")),
                   Person(PersonLabels(email_label="c"))]

    meta = MetaExtractor().extract(df, annotations=annotations)

    assert hasattr(meta, "_df_meta")
    assert isinstance(meta._df_meta["Bank_a"], _Bank)
    assert isinstance(meta._df_meta["Address_b"], _Address)
    assert isinstance(meta._df_meta["Person_c"], _Person)


def test_type_override(df_meta):
    overrides = [ContinuousModel("a", df_meta), DiscreteModel("b", df_meta)]

    synthesizer = HighDimSynthesizer(df_meta, type_overrides=overrides)

    assert isinstance(synthesizer._synthesizer.df_model["a"], _ContinuousModel)
    assert isinstance(synthesizer._synthesizer.df_model["b"], _DiscreteModel)
