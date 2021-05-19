import numpy as np
import pandas as pd
import pytest

from synthesized import MetaExtractor
from synthesized.complex.unifier.ensemble import EnsembleUnifier, _resolve_overlap
from synthesized.model import factory
from synthesized.model.models import KernelDensityEstimate


def test_resolve_overlap():
    df = pd.DataFrame({"A": [0, 1, 2, 3], "cat_overlap": [0.1, 0.6, -1.1, 100], "cont_overlap": [0.1, -0.2, -1.1, 100]})
    df_ = pd.DataFrame({"B": [0, 1, 2, 3], "cat_overlap": [0.6, -1.1, 100, 0], "cont_overlap": [-0.21, -2, 100, 100]})

    df_meta = MetaExtractor.extract(df_)
    # force continuous column to have continuous model
    cont_model = KernelDensityEstimate(df_meta["cont_overlap"])
    df_model = factory.ModelFactory()(df_meta, type_overrides=[cont_model])

    df_out = _resolve_overlap(df, df_, df_model)

    df_expected = pd.DataFrame(
        {"A": [1, 3], "B": [0, 2], "cat_overlap": [0.6, 100], "cont_overlap": [-0.2, 100]}
    )

    assert df_out.equals(df_expected)


@pytest.fixture(scope="module")
def ensemble_unifier():
    df0 = pd.DataFrame({"A": np.arange(100), "overlap": np.random.choice([0, 1], size=100)})
    df1 = pd.DataFrame({"B": np.arange(100), "overlap": np.random.choice([0, 1], size=100)})

    df0_meta = MetaExtractor.extract(df0)
    df1_meta = MetaExtractor.extract(df1)

    unifier = EnsembleUnifier()
    unifier.update(df0_meta, df0, num_iterations=10)
    unifier.update(df1_meta, df1, num_iterations=10)
    return unifier


@pytest.mark.slow
@pytest.mark.parametrize("A", [False, True])
@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("B", [False, True])
def test_query(ensemble_unifier: EnsembleUnifier, A, overlap, B):
    columns = (["A"] if A else []) + (["overlap"] if overlap else []) + (["B"] if B else [])
    df_out: pd.DataFrame = ensemble_unifier.query(columns=columns, num_rows=100)

    assert df_out.columns.equals(pd.Index(columns))
    assert len(df_out) == 100
    with pytest.raises(ValueError):
        ensemble_unifier.query(columns=["fake_column"], num_rows=100)
