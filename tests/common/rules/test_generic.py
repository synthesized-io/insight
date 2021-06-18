import pandas as pd
import pytest

from synthesized.common.rules import CaseWhenThen, ValueEquals, ValueIsIn, ValueRange


def test_range():

    df = pd.DataFrame(dict(x=[0, 1, 2, 3, 4, 5], y=[5, 4, 3, 2, 1, 0]))

    with pytest.raises(ValueError):
        ValueRange("x", low=10, high=0)

    # 0 <= x <= 5
    r = ValueRange("x", low=0, high=5, low_inclusive=True, high_inclusive=True)
    assert r._to_str() == "0 <= x & 5 >= x"
    assert r._is_valid(df).sum() == 6

    # 0 < x < 5
    r = ValueRange("x", low=0, high=5, low_inclusive=False, high_inclusive=False)
    assert r._to_str() == "0 < x & 5 > x"
    assert r._is_valid(df).sum() == 4

    # x <= y
    r = ValueRange("x", high="y", high_inclusive=True)
    assert r._to_str() == "-inf < x & y >= x"
    assert r._is_valid(df).sum() == 3

    # x >= y
    r = ValueRange("x", low="y", low_inclusive=True)
    assert r._to_str() == "y <= x & inf > x"
    assert r._is_valid(df).sum() == 3


def test_value_equals():
    df = pd.DataFrame(dict(x=[0, "A", 2, 3, 4, 5], y=[0, "B", 2, 3, 4, 5]))

    # x == 3
    r = ValueEquals("x", 3)
    assert r._is_valid(df).sum() == 1

    # x == y
    r = ValueEquals("x", "y")
    assert r._is_valid(df).sum() == 5

    # x == 'A'
    r = ValueEquals("x", "'A'")
    assert r._is_valid(df).sum() == 1


def test_value_is_in():
    df = pd.DataFrame(dict(x=[0, "A", 2, 3, 4, 5], y=[0, "B", "C", 3, 4, 5]))

    # x in [3, 2]
    r = ValueIsIn("x", [3, 2])
    assert r._is_valid(df).sum() == 2

    # y in ["B", "C"]
    r = ValueIsIn("y", ["B", "C"])
    assert r._is_valid(df).sum() == 2

    # x in [1]
    r = ValueIsIn("x", [1])
    assert r._is_valid(df).sum() == 0


def test_case_when_then():

    df = pd.DataFrame(dict(x=[0, 1, 2, 3, 4, 5], y=[5, 4, 3, 2, 1, 0]))

    with pytest.raises(ValueError):
        CaseWhenThen(ValueRange("x", 0, 10), ValueRange("x", 0, 10))

    # when 0 <= x <= 5 then when 0 <= y <= 5
    r = CaseWhenThen(
        ValueRange("x", 0, 5, low_inclusive=True, high_inclusive=True),
        ValueRange("y", 0, 5, low_inclusive=True, high_inclusive=True),
    )
    assert r._is_valid(df).sum() == 6

    # when 0 <= x <= 5 then y > 10
    r = CaseWhenThen(ValueRange("x", 0, 5, low_inclusive=True, high_inclusive=True), ValueRange("y", low=10))
    assert r._is_valid(df).sum() == 0

    # when x < 3 then y > x
    r = CaseWhenThen(ValueRange("x", high=3), ValueRange("y", low="x"))
    assert r._is_valid(df).sum() == 6

    # when 0 <= x <= 5 then y > x
    r = CaseWhenThen(ValueRange("x", 0, 5, low_inclusive=True, high_inclusive=True), ValueRange("y", low="x"))
    assert r._is_valid(df).sum() == 3


def test_filter():

    df = pd.DataFrame(dict(x=[0, 1, 2, 3, 4, 5], y=[5, 4, 3, 2, 1, 0]))
    r = ValueRange("x", 0, 2, low_inclusive=True, high_inclusive=True)
    pd.testing.assert_frame_equal(pd.DataFrame(dict(x=[0, 1, 2], y=[5, 4, 3])), r.filter(df))
