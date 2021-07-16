
import pandas as pd
import pytest

from synthesized.suite import JoinStatement, RuleStatement, TestingSuite

from .cases import case_nn, case_synthetic, case_tpch


@pytest.mark.parametrize(
    "case,expected",
    [(case_tpch, {}), (case_synthetic, {}), (case_nn, {})],
)
def test_suite_suite_manager_acceptance(case, expected):
    selector, rule_stmt_df, join_stmt_df = case()

    rule_list = [
        RuleStatement(target=target, raw_stmt=rule_stmt_raw)
        for _, (target, rule_stmt_raw) in rule_stmt_df.iterrows()
    ]
    join_list = [
        JoinStatement(target=target, raw_stmt=join_stmt_raw)
        for _, (target, join_stmt_raw) in join_stmt_df.iterrows()
    ]

    suite = TestingSuite(
        selector=selector,
        rule_list=rule_list,
        join_list=join_list,
    )

    for result in suite.generate():
        print(result)
        assert result.n_analyzed_rules == result.n_total_rules
        assert result.coverage_new.coverage == 1
        assert result.coverage_new.samples <= result.n_analyzed_rules
