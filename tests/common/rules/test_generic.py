import pandas as pd
import pytest

# from synthesized.common.rules import CaseWhenThen, ValueEquals, ValueIsIn, ValueRange
from synthesized.common.rules import And, CaseWhen, Column, Equals, GreaterThan, IsIn, LowerThan, Value, ValueRange


def test_range():

    df = pd.DataFrame(dict(x=[0, 1, 2, 3, 4, 5], y=[5, 4, 3, 2, 1, 0]))

    # 0 <= x <= 5
    r = ValueRange(Column("x"), [Value(0), Value(5)], low_inclusive=True, high_inclusive=True)
    assert r.to_pd_str("df") == "((0 <= df['x']) & (df['x'] <= 5))"
    assert r._is_valid(df).sum() == 6

    # 0 < x < 5
    r = ValueRange(Column("x"), [Value(0), Value(5)], low_inclusive=False, high_inclusive=False)
    assert r.to_pd_str("df") == "((0 < df['x']) & (df['x'] < 5))"
    assert r._is_valid(df).sum() == 4

    # x <= y
    r = LowerThan(Column("x"), Column("y"), inclusive=True)
    assert r.to_pd_str("df") == "(df['x'] <= df['y'])"
    assert r._is_valid(df).sum() == 3

    # x >= y
    r = GreaterThan(Column("x"), Column("y"), inclusive=True)
    assert r.to_pd_str("df") == "(df['x'] >= df['y'])"
    assert r._is_valid(df).sum() == 3


def test_value_equals():
    df = pd.DataFrame(dict(x=[0, "A", 2, 3, 4, 5], y=[0, "B", 2, 3, 4, 5]))

    # x == 3
    r = Equals(Column("x"), Value(3))
    assert r._is_valid(df).sum() == 1

    # x == y
    r = Equals(Column("x"), Column("y"))
    assert r._is_valid(df).sum() == 5

    # x == 'A'
    r = Equals(Column("x"), Value("A"))
    assert r._is_valid(df).sum() == 1


def test_value_is_in():
    df = pd.DataFrame(dict(x=[0, "A", 2, 3, 4, 5], y=[0, "B", "C", 3, 4, 5]))

    # x in [3, 2]
    r = IsIn(Column("x"), [Value(3), Value(2)])
    assert r._is_valid(df).sum() == 2

    # y in ["B", "C"]
    r = IsIn(Column("y"), [Value("B"), Value("C")])
    assert r._is_valid(df).sum() == 2

    # x in [1]
    r = IsIn(Column("x"), [Value(1)])
    assert r._is_valid(df).sum() == 0


def test_and():
    df = pd.DataFrame(dict(x=[0, 1, 2, 3, 4, 5], y=[5, 4, 3, 2, 1, 0]))

    # 0 <= x <= 5 and 0 <= y <= 5
    r = And([
        ValueRange(Column("x"), [Value(0), Value(5)], low_inclusive=True, high_inclusive=True),
        ValueRange(Column("y"), [Value(0), Value(5)], low_inclusive=True, high_inclusive=True),
    ])
    assert r._is_valid(df).sum() == 6

    # 0 <= x <= 5 and y > 10
    r = And([
        ValueRange(Column("x"), [Value(0), Value(5)], low_inclusive=True, high_inclusive=True),
        GreaterThan(Column("y"), Value(10))
    ])
    assert r._is_valid(df).sum() == 0

    # x < 3 and y > x
    r = And([
        LowerThan(Column("x"), Value(3)),
        GreaterThan(Column("y"), Column("x"))
    ])
    assert r._is_valid(df).sum() == 3

    # 0 <= x <= 5 and y > x
    r = And([
        ValueRange(Column("x"), [Value(0), Value(5)], low_inclusive=True, high_inclusive=True),
        GreaterThan(Column("y"), Column("x"))
    ])
    assert r._is_valid(df).sum() == 3


def test_filter():

    df = pd.DataFrame(dict(x=[0, 1, 2, 3, 4, 5], y=[5, 4, 3, 2, 1, 0]))
    r = ValueRange(Column("x"), [Value(0), Value(2)], low_inclusive=True, high_inclusive=True)
    pd.testing.assert_frame_equal(pd.DataFrame(dict(x=[0, 1, 2], y=[5, 4, 3])), r.filter(df))
