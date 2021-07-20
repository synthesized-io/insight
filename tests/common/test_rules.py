import random
import string

import numpy as np
import pandas as pd
import pytest

from synthesized import HighDimSynthesizer
from synthesized.common.rules import Column, Equals, Expression, Function, IsIn, Value
from synthesized.metadata.factory import MetaExtractor
from tests.utils import progress_bar_testing


@pytest.mark.slow
@pytest.mark.parametrize("use_generic_rule", [False, True])
@pytest.mark.parametrize("use_expression_rule", [False, True])
def test_synthesize_with_rules(use_generic_rule, use_expression_rule):
    size = 1000

    df_original = pd.DataFrame({
        'x1': np.random.normal(loc=0, scale=1, size=size),
        'x2': np.random.normal(loc=0, scale=1, size=size),
        'y1': np.random.choice(['A', 'B'], size=size),
        'y2': np.random.choice(['A', 'B'], size=size),
        'sample': [''.join([random.choice(string.ascii_letters) for _ in range(10)]) for _ in range(size)]
    })

    df_meta = MetaExtractor.extract(df=df_original)
    synthesizer = HighDimSynthesizer(df_meta=df_meta)
    synthesizer.learn(num_iterations=10, df_train=df_original)

    generic_rules = [IsIn(Column("y1"), [Value("A")])] if use_generic_rule else []
    expression_rules = [Expression(Column("x3"), "x1 + x2")] if use_expression_rule else []
    df_synthesized = synthesizer.synthesize_from_rules(num_rows=len(df_original), progress_callback=progress_bar_testing,
                                                       generic_rules=generic_rules, expression_rules=expression_rules)

    if use_expression_rule:
        assert (df_synthesized["x3"] == df_synthesized["x1"] + df_synthesized["x2"]).all()
    if use_generic_rule:
        assert (df_synthesized["y1"] == "A").all()

    assert len(df_synthesized) == len(df_original)


def test_synthesize_bad_generic_rules_raises_error():
    size = 1000

    df_original = pd.DataFrame({
        'y1': np.random.choice(['A', 'B'], size=size),
    })

    df_meta = MetaExtractor.extract(df=df_original)
    synthesizer = HighDimSynthesizer(df_meta=df_meta)
    synthesizer.learn(num_iterations=10, df_train=df_original)

    generic_rules = [IsIn(Column("y1"), [Value("C")])]
    with pytest.raises(RuntimeError):
        synthesizer.synthesize_from_rules(num_rows=len(df_original), generic_rules=generic_rules)


@pytest.fixture
def df():
    return pd.DataFrame(dict(
        x=["a", "abc", "ABC", "ckld", "abcnmca", "mcdaabc", "mabcmewq", "aa"],
        y=["2010-12-12", "2011-12-12", "2010-12-12", "2012-12-12", "2010-12-12", "2010-12-12", "2010-12-12", "2010-12-12"],
        z=["2010-12-12", "aa", "2010-12-12", "bb", "2010-12-12", "cc", "2010-12-12", "dd"]
    ))


@pytest.mark.parametrize("inverse", [True, False])
@pytest.mark.parametrize(
    "func_name,args,value,expected_len",
    [
        ("LEN", [Column("x")], 3, 2),
        ("UPPER", [Column("x")], "ABC", 2),
        ("LOWER", [Column("x")], "abc", 2),
        ("RIGHT", [Column("x"), Value(3)], "abc", 2),
        ("LEFT", [Column("x"), Value(3)], "abc", 2),
        ("SUBSTRING", [Column("x"), Value(1), Value(4)], "abc", 1),
        ("CONCAT", [Column("x"), Column("x")], "aaaa", 1),
        ("DATEDIFF", [Value("MONTH"), Column("y"), Column("y")], 0, 8),
        ("IS_DATE", [Column("z")], True, 4),
    ]
)
def test_rule_functions(df, func_name, args, value, expected_len, inverse):
    func = Function.from_name(func_name, arguments=args)
    eq = Equals(func, Value(value))

    assert len(eq.filter(df)) == expected_len
    funcs = eq.get_augment_func(inverse=inverse)

    for col_name, func in funcs.items():
        df_idx = df.sample(1).copy()
        df_idx[col_name] = df_idx.apply(func, axis=1)
