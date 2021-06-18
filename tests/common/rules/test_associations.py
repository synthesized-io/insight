import numpy as np
import pandas as pd
import pytest

from synthesized import MetaExtractor
from synthesized.common.rules import Association
from synthesized.common.rules.exceptions import AssociatedRuleOverlap


def test_associations():

    df = pd.DataFrame({
        'a': [0, 0, 1, 1],
        'b': [0, 1, 0, 1],
        'c': [0, 0, 0, 1]
    })

    df_meta = MetaExtractor.extract(df)
    rule = Association.detect_association(df=df, df_meta=df_meta, associations=['a', 'b', 'c'])

    true_binding_mask = np.array([[[1, 0], [1, 0]], [[1, 0], [0, 1]]])
    np.testing.assert_array_almost_equal(rule.binding_mask, true_binding_mask)


def test_association_nans():
    df = pd.DataFrame({
        'a': [0, 0, 1, 1],
        'b': [0, 1, np.nan, 1],
        'c': [np.nan, np.nan, 0, np.nan]
    })

    df_meta = MetaExtractor.extract(df)
    rule = Association.detect_association(df=df, df_meta=df_meta, associations=['a'], nan_associations=['b', 'c'])

    true_binding_mask = np.array(([[[0, 1], [0, 0]], [[0, 1], [1, 0]]]))
    np.testing.assert_array_almost_equal(rule.binding_mask, true_binding_mask)


def test_raises_error_on_nan_association():
    with pytest.raises(AssociatedRuleOverlap):
        Association(np.array([]), associations=["a"], nan_associations=["a"])


def test_repeated_columns_raises_error_on_validate():
    association_1 = Association(binding_mask=np.array([]), associations=["a"])
    association_2 = Association(binding_mask=np.array([]), associations=["a"])

    with pytest.raises(AssociatedRuleOverlap):
        Association._validate_association_rules([association_1, association_2])
