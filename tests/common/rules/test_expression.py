import pandas as pd

from synthesized.common.rules import Column, Expression


def test_expression():
    df = pd.DataFrame(dict(x=[1, 2, 3], y=[0, 1, 1]))

    r = Expression(Column("z"), "x + 2*y")
    assert r.apply(df)["z"].values.tolist() == [1, 4, 5]
