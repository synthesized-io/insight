from dataclasses import dataclass, field
from typing import Dict, List, Optional

from dataclasses_json import DataClassJsonMixin

from .helpers import replace_alias_with_table_name
from .parsers import ParsedJoinStmt, ParsedRuleStmt, parse_join_stmt, parse_rule_stmt


@dataclass
class JoinStatement(DataClassJsonMixin):
    target: str
    raw_stmt: str
    _parsed_stmt: Optional[ParsedJoinStmt] = None

    @property
    def parsed_stmt(self) -> ParsedJoinStmt:
        if self._parsed_stmt is None:
            self._parsed_stmt = parse_join_stmt(self.raw_stmt)
        return self._parsed_stmt


@dataclass
class RuleStatement(DataClassJsonMixin):
    target: str
    raw_stmt: str
    _parsed_stmt: Optional[ParsedRuleStmt] = None

    def replace_alias_in_raw_stmts(self, alias_to_table_name: Dict[str, str]):
        self.raw_stmt = replace_alias_with_table_name(
            self.raw_stmt, alias_to_table_name
        )
        if self._parsed_stmt is not None:
            self._parsed_stmt = parse_rule_stmt(self.raw_stmt)

    @property
    def parsed_stmt(self) -> ParsedRuleStmt:
        if self._parsed_stmt is None:
            self._parsed_stmt = parse_rule_stmt(self.raw_stmt)
        return self._parsed_stmt


@dataclass
class CoverageResult:
    coverage: Optional[float] = None
    samples: Optional[int] = None
    not_covered_rules: Optional[List[str]] = None

    def __repr__(self) -> str:
        return (
            f"CoverageResult(coverage={self.coverage}, samples={self.samples}, "
            f"n_not_covered_rules={len(self.not_covered_rules) if self.not_covered_rules else 0}"
        )


@dataclass
class RuleResult:
    rule_target: str
    rule_raw: str
    coverage_old: Optional[float] = None
    coverage_new: Optional[float] = None
    number_of_subrules: Optional[int] = None
    analyzed_subrules: int = 0


@dataclass
class TestingSuiteResult(DataClassJsonMixin):
    target: str
    data: Dict[str, dict]
    coverage_old: CoverageResult
    coverage_new: CoverageResult
    n_analyzed_rules: int
    n_total_rules: int
    rule_result_list: List[RuleResult] = field(default_factory=lambda: [])

    def __repr__(self) -> str:
        return (
            "TestingSuiteResult("
            f"\n\ttarget='{self.target}', "
            f"\n\tcoverage_old={self.coverage_old}, "
            f"\n\tcoverage_new={self.coverage_new}, "
            f"\n\tn_analyzed_rules={self.n_analyzed_rules}, "
            f"\n\tn_total_rules={self.n_total_rules}\n)"
        )
