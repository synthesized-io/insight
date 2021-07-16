
from typing import Dict, Iterator, List

import numpy as np
import pandas as pd

from synthesized.suite.engine import augment_df, compute_data_coverage, compute_rule_coverage, get_optimal_df

from .exceptions import ParserError
from .selector import SelectedData, Selector, selected_data_to_dict
from .types import JoinStatement, RuleResult, RuleStatement, TestingSuiteResult


class TestingSuite:
    def __init__(self, selector: Selector, join_list: List[JoinStatement], rule_list: List[RuleStatement]):
        self.selector = selector
        self.join_list = join_list
        self.rule_list = rule_list

        # Filter out NaN rules
        self.rule_list = [
            r for r in self.rule_list
            if not (isinstance(r.raw_stmt, float) and np.isnan(r.raw_stmt))
        ]
        if len(self.rule_list) == 0:
            raise ValueError("Given statements column in rule_list is empty.")

        # Replace alias in rules
        for join in self.join_list:
            for rule in self.rule_list:
                if rule.target != join.target:
                    continue

                rule.replace_alias_in_raw_stmts(join.parsed_stmt.alias_to_table_name)

    def _collect_target_to_rule_list(self) -> Dict[str, List[RuleStatement]]:
        """Collect all rule statements referring to one target"""
        target_to_rule_list: Dict[str, List[RuleStatement]] = {}
        for rule in self.rule_list:
            if rule.target not in target_to_rule_list:
                target_to_rule_list[rule.target] = []
            if rule.raw_stmt and isinstance(rule.raw_stmt, str):  # Remove NaNs from statements
                target_to_rule_list[rule.target].append(rule)

        return target_to_rule_list

    def generate(self) -> Iterator[TestingSuiteResult]:
        """
        Default entry point.
        """
        # collect all rule statements referring to one target
        target_to_rule_list = self._collect_target_to_rule_list()

        # compute result for each target
        for join in self.join_list:
            joined_data: pd.DataFrame = self.selector.join(stmt=join.parsed_stmt)

            alias_to_table_name = join.parsed_stmt.alias_to_table_name
            rule_list = target_to_rule_list.get(join.target, [])

            joined_data_new: pd.DataFrame = pd.DataFrame()
            n_analyzed_rules = 0
            rule_result_list: List[RuleResult] = []
            for rule in rule_list:
                rule.replace_alias_in_raw_stmts(alias_to_table_name)
                rule_result = RuleResult(
                    rule_target=rule.target, rule_raw=rule.raw_stmt
                )
                try:
                    parsed_stmt = rule.parsed_stmt
                except ParserError:
                    # can't be parsed -- default values
                    rule_result_list.append(rule_result)
                    continue

                # augment df according to the rule
                augmented_df = augment_df(df=joined_data, stmt=parsed_stmt)
                n_analyzed_rules += len(parsed_stmt)
                if len(joined_data_new) == 0:
                    joined_data_new = augmented_df
                else:
                    joined_data_new = pd.concat(
                        [joined_data_new, augmented_df], ignore_index=True
                    )

                # compute rule's coverage
                rule_result.coverage_old = compute_rule_coverage(
                    df=joined_data,
                    rule=rule,
                )
                rule_result.coverage_new = compute_rule_coverage(
                    df=joined_data_new,
                    rule=rule,
                )
                rule_result.number_of_subrules = len(parsed_stmt)
                if rule_result.coverage_new == 0:
                    rule_result.analyzed_subrules = 0
                else:
                    rule_result.analyzed_subrules = int(rule_result.number_of_subrules * rule_result.coverage_old
                                                        * (1.0 / rule_result.coverage_new))
                rule_result_list.append(rule_result)

            joined_data_new = get_optimal_df(df=joined_data_new, rule_list=rule_list)

            coverage_old = compute_data_coverage(
                df=joined_data,
                rule_list=rule_list,
            )
            coverage_new = compute_data_coverage(
                df=joined_data_new, rule_list=rule_list, warning_not_covered=True
            )
            unjoined_data: SelectedData = self.selector.unjoin(
                df=joined_data_new, stmt=join.parsed_stmt
            )
            yield TestingSuiteResult(
                target=join.target,
                data=selected_data_to_dict(unjoined_data),
                coverage_old=coverage_old,
                coverage_new=coverage_new,
                n_analyzed_rules=n_analyzed_rules,
                n_total_rules=sum([len(rule.parsed_stmt) for rule in rule_list]),
                rule_result_list=rule_result_list,
            )
