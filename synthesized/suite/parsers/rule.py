from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Union, cast

from lark import Token, Transformer, Tree

from .grammar import load_grammar
from ...common.rules import (And, Column, Equals, Function, GenericRule, GreaterThan, IsIn, IsNull, LogicComparator,
                             LowerThan, Negate, Not, Operation, Or, TableColumn, UnhappyFlowNode, Value)

ParsedRuleStmt = List[Tuple[GenericRule, GenericRule]]

rule_lr = load_grammar(start="expr")


def get_operator_str(operator_sign: str):
    if operator_sign == "+":
        return "_add"
    elif operator_sign == "-":
        return "_minus"
    elif operator_sign == "*":
        return "_prod"
    elif operator_sign == "/":
        return "_div"
    else:
        raise KeyError(f"Given operator '{operator_sign}' not recognized")


@dataclass
class TransformedExpr:
    """CASE [WHEN pred THEN value] for pred, value in zip(pred_list, value_list)"""

    pred_list: Optional[List[GenericRule]]
    value_list: List[GenericRule]

    def __post_init__(self):
        if self.pred_list:
            for pred in self.pred_list:
                assert isinstance(pred, GenericRule), f"type(val) = {type(pred)}, pred={pred}"
            assert len(self.pred_list) == len(self.value_list)
        else:
            assert len(self.value_list) == 1
        for val in self.value_list:
            assert isinstance(val, GenericRule), f"type(val) = {type(val)}"

    def to_list(self) -> Iterator[Tuple[Optional[GenericRule], GenericRule]]:
        if self.pred_list is None:
            yield (None, self.value_list[0])
            return

        for pred, value in zip(self.pred_list, self.value_list):
            yield (pred, value)

    def apply_to_decart_multiple(self, other: "TransformedExpr", predicate_callable: Callable,
                                 value_callable: Callable) -> "TransformedExpr":
        pred_list = []
        value_list = []

        if self.is_pred_empty or other.is_pred_empty:
            return self.apply_to_decart_multiple_pred_empty(other=other, predicate_callable=predicate_callable,
                                                            value_callable=value_callable)

        for self_pred, self_result in self.to_list():
            for other_pred, other_result in other.to_list():
                pred_list.append(predicate_callable(self_pred, other_pred) if self_pred and other_pred else None)
                value_list.append(value_callable(self_result, other_result))

        return TransformedExpr(pred_list=pred_list, value_list=value_list)

    def apply_to_decart_multiple_pred_empty(self, other: "TransformedExpr", predicate_callable: Callable,
                                            value_callable: Callable) -> "TransformedExpr":
        """If one of the predicates is pred_empty."""
        pred_list = []
        value_list = []

        if self.is_pred_empty and other.is_pred_empty:
            return TransformedExpr(pred_list=None,
                                   value_list=[value_callable(self.value_list[0], other.value_list[0])])

        if self.is_pred_empty and not other.is_pred_empty:
            for other_pred, other_result in other.to_list():
                assert other_pred and self.value_list[0] and other_result
                pred_list.append(other_pred)
                value_list.append(value_callable(self.value_list[0], other_result))
            return TransformedExpr(pred_list=pred_list, value_list=value_list)

        if not self.is_pred_empty and other.is_pred_empty:
            return other.apply_to_decart_multiple(self, predicate_callable=predicate_callable,
                                                  value_callable=value_callable)

        raise ValueError

    def __add__(self, other: "TransformedExpr") -> "TransformedExpr":
        return self._operation(other, operation="_add")

    def _operation(self, other: "TransformedExpr", operation: str) -> "TransformedExpr":
        def add_pred(
            d1: Optional[GenericRule], d2: Optional[GenericRule]
        ) -> Optional[GenericRule]:
            if d1 is None and d2 is not None:
                return d2
            elif d2 is None and d1 is not None:
                return d1
            if d1 is None and d2 is None:
                return None
            assert isinstance(d1, GenericRule)
            assert isinstance(d2, GenericRule)
            return And([d1, d2])

        return self.apply_to_decart_multiple(
            other, predicate_callable=add_pred,
            value_callable=lambda r1, r2: Operation.from_operator(operation, r1, r2) if r1 and r2 else None
        )

    def to_value(self) -> GenericRule:
        if not self.is_pred_empty:
            raise ValueError(f"{self.value_list}")
        return self.value_list[0]

    @property
    def is_pred_empty(self):
        return self.pred_list is None and len(self.value_list) == 1

    @staticmethod
    def from_value(s: GenericRule) -> "TransformedExpr":
        return TransformedExpr(pred_list=None, value_list=[s])


@dataclass
class TransformedPredicate:
    pred: Optional[GenericRule]

    def _operation(self, atom: GenericRule, operation: str):
        if self.pred is None:
            return TransformedPredicate(pred=atom)
        if atom is None:
            return self

        return Operation.from_operator(operation, self.pred, atom)

    def _or(self, atom: dict):
        if self.pred is None or len(atom) == 0:
            return TransformedPredicate(pred=None)
        return TransformedPredicate(pred=Or([self.pred, atom]))

    def to_dict(self):
        return self.pred


class RuleTransformer(Transformer):
    def __init__(self, columns: List[str] = None, new_rules: Dict[str, Callable] = None) -> None:
        self.columns = columns
        self.new_rules = new_rules if new_rules else {}
        super().__init__()

    def strip_lookup_varnamew(self, s):  # ALIAS.COLUMN_NAME
        table_name = s[0].value.lower()
        column_name = s[1].value.lower()
        return TableColumn(column_name=column_name, table_name=table_name)

    def str_varnamew(self, s):
        (s,) = s
        return Value(s)

    def varnamew(self, s):  # COLUMN_NAME
        (s,) = s
        return TableColumn(column_name=s.lower())

    def tab_name(self, s):
        return s[0]

    def col_name(self, s):
        return s[0]

    def arguments(self, s):
        return s  # proxy to self.function

    def _recursive_function(self, funcname: Value, arguments: List[TransformedExpr]):
        # If it's a function of a function, we want to apply the first function to the value of the second one

        transf_expr = arguments[0]  # not is_pred_empty
        fargs = arguments[1:]

        pred_list: List[GenericRule] = []
        value_list: List[GenericRule] = []

        for pred, value in transf_expr.to_list():
            assert pred is not None
            if isinstance(value, UnhappyFlowNode):
                pred_list.append(pred)
                value_list.append(value)
                continue

            func_out = self.function((funcname, [value] + fargs))  # type: ignore
            if isinstance(func_out, Function):
                pred_list.append(pred)
                value_list.append(func_out)
            else:
                assert isinstance(func_out, TransformedExpr) and func_out.pred_list
                for func_pred_i in func_out.pred_list:
                    pred_list.append(And([pred, func_pred_i]))
                value_list.extend(func_out.value_list)

        return pred_list, value_list

    def func_datediff(self, s):
        s = [Value(s[0]), s[1], s[2]]
        return self.function((Value("DATEDIFF"), s))

    def function(self, s: Tuple[Value, List]) -> Union[Function, TransformedExpr]:
        funcname, arguments = s

        if (isinstance(arguments, (list, tuple)) and len(arguments) >= 1
                and isinstance(arguments[0], TransformedExpr) and not arguments[0].is_pred_empty):
            assert all(arg.is_pred_empty for arg in arguments[1:])
            pred_list, value_list = self._recursive_function(funcname, arguments)
            return TransformedExpr(pred_list=pred_list, value_list=value_list)

        if not isinstance(arguments, (list, tuple)):
            arguments = [arguments]

        assert isinstance(funcname.value, str)
        arguments = [arg.to_value() if isinstance(arg, TransformedExpr) else arg for arg in arguments]
        arguments = [v.value if isinstance(v, Token) else v for v in arguments]

        if funcname.value in self.new_rules.keys():
            return self.new_rules[funcname.value](arguments)

        if funcname.value.lower() == "isnull":
            return TransformedExpr(
                pred_list=[Not(IsNull(arguments[0])), IsNull(arguments[0])],
                value_list=[arguments[0], arguments[1]],
            )

        if funcname.value.lower() == "coalesce":
            pred_list = []
            value_list = []

            col_arguments = [arg for arg in arguments if len(get_all_cols(arg)) > 0]
            for i, col in enumerate(col_arguments):
                and_pred: List[GenericRule] = [IsNull(prev_col) for prev_col in col_arguments[:i]]
                and_pred.append(Not(IsNull(col)))
                pred_list.append(And(and_pred))
                value_list.append(col)

            pred_list.append(And([IsNull(col) for col in col_arguments]))
            value_list.append(UnhappyFlowNode())

            return TransformedExpr(pred_list=pred_list, value_list=value_list)

        return Function.from_name(func_name=funcname.value, arguments=arguments)

    def func_name(self, s):
        return s[0]

    def variable(self, s):
        return s[0]

    def string(self, s):
        return Value(s[0][1:-1])

    def number(self, s):
        return Value(float(s[0]))

    def negative_number(self, s):
        return Value(-float(s[1]))

    def value(self, s):
        return s[0]

    def value_null(self, s):
        return Value(None)

    def casted_value(self, s):
        return s[0]

    def casted_expr(self, s):
        return s[0]

    def expr_case_add(self, s) -> TransformedExpr:
        (expr0, operator, expr1) = s

        assert isinstance(expr0, TransformedExpr) and expr0.pred_list, expr0
        assert isinstance(expr1, TransformedExpr) and expr1.pred_list, expr1
        assert isinstance(operator, Tree), operator

        return TransformedExpr(
            pred_list=expr0.pred_list + expr1.pred_list,
            value_list=expr0.value_list + expr1.value_list,
        )

    def operation_expr(self, s) -> TransformedExpr:
        (expr0, operator, expr1) = s

        assert isinstance(expr0, TransformedExpr), expr0
        assert isinstance(expr1, TransformedExpr), expr1
        assert isinstance(operator, Tree), operator

        operator_str = get_operator_str(operator.children[0].value)  # type: ignore
        return expr0._operation(expr1, operation=operator_str)

    def neg_expr(self, s) -> TransformedExpr:
        (expr, operator) = s
        assert isinstance(expr, TransformedExpr), expr

        expr.value_list[0] = Negate(expr.value_list[0])

        return expr

    def value_to_expr(self, s) -> TransformedExpr:
        (s,) = s
        if isinstance(s, TransformedExpr):
            return s
        return TransformedExpr.from_value(s)

    def pred_atom_compare_value(self, s):
        return self._atom_pred_to_dict(s[0], s[1], s[2])

    def pred_atom_in_list(self, s):
        return self._atom_pred_to_dict(s[0], "in", s[2:])

    def pred_atom_not_in_list(self, s):
        return self._atom_pred_to_dict(s[0], "not in", s[2:])

    def pred_atom_is_null(self, s):
        return self._atom_pred_to_dict(s[0], "is", None)

    def pred_atom_is_not_null(self, s):
        return self._atom_pred_to_dict(s[0], "is not", None)

    def pred_atom_like_value(self, s):
        return self._atom_pred_to_dict(s[0], "like", s[1])

    def _atom_pred_to_dict(self, expr: TransformedExpr, comparator, if_value):
        """WHEN [expr] [comparator] [if_value] THEN"""

        if not expr.is_pred_empty:
            result = []
            for pred, value in expr.to_list():
                re_value = self._atom_pred_to_dict(TransformedExpr.from_value(value), comparator, if_value)
                if re_value:
                    expr_i = And([pred, re_value])
                    result.append(expr_i)
            return result

        expr_result = expr.to_value()
        expr_result = expr_result.value if isinstance(expr_result, Token) else expr_result

        if isinstance(expr_result, UnhappyFlowNode):
            return UnhappyFlowNode()

        return self._comparator_to_rule(comparator, expr_result, if_value)

    @staticmethod
    def _comparator_to_rule(comparator, expr_result, if_value):
        if comparator.lower() in ["=", "like"]:
            return Equals(expr_result, if_value)
        elif comparator in ["!=", "<>"]:
            return Not(Equals(expr_result, if_value))
        elif comparator.lower() == "in":
            return IsIn(expr_result, [Value(val.value) for val in if_value])
        elif comparator.lower() == "not in":
            return Not(IsIn(expr_result, [Value(val.value) for val in if_value]))
        elif comparator.lower() == "is":
            assert if_value is None
            return IsNull(expr_result)
        elif comparator.lower() == "is not":
            assert if_value is None
            return Not(IsNull(expr_result))
        elif comparator == "<":
            return LowerThan(expr_result, if_value, inclusive=False)
        elif comparator == "<=":
            return LowerThan(expr_result, if_value, inclusive=True)
        elif comparator == ">":
            return GreaterThan(expr_result, if_value, inclusive=False)
        elif comparator == ">=":
            return GreaterThan(expr_result, if_value, inclusive=True)
        else:
            raise ValueError(f"unknown comparator {comparator}")

    def bypass_parenthesis(self, s):
        return s[0]

    def predicate_complex_and(self, s):
        atom, pred = s
        assert isinstance(pred, TransformedPredicate), type(atom)
        assert isinstance(pred, TransformedPredicate), type(pred)
        return TransformedPredicate(And([pred.pred, atom.pred]))

    def predicate_complex_or(self, s):
        atom, pred = s
        assert isinstance(pred, TransformedPredicate), type(atom)
        assert isinstance(pred, TransformedPredicate), type(pred)
        return TransformedPredicate(Or([pred.pred, atom.pred]))

    def predicate_complex_atom(self, s):
        (atom,) = s

        if isinstance(atom, GenericRule):
            return TransformedPredicate(pred=atom)
        else:
            return [TransformedPredicate(pred=atom_i) for atom_i in atom if atom_i]

    def predicate(self, s):
        return s[0]

    def case_item_expr_homogeneous(self, s):
        when_expr, then_expr = s
        assert isinstance(then_expr, TransformedExpr), f"{then_expr} {type(then_expr)}"

        if isinstance(when_expr, list):
            # If we have multible predicates for same 'then_expr'

            when_exprs = []
            then_exprs = []
            for when_expr_i in when_expr:
                when_exprs.append(when_expr_i)
                then_exprs.append(then_expr)

            return when_exprs, then_exprs

        assert isinstance(when_expr, TransformedExpr), f"{when_expr} {type(when_expr)}"
        return when_expr.to_value(), then_expr

    def case_expr_homogeneous(self, s):
        case_expr, homog_case_item_list, else_expr = s[0], s[1:-1], s[-1]
        return self._case_item_to_expr(
            [
                (TransformedPredicate(self._atom_pred_to_dict(case_expr, "=", when_value)), then_expr)
                for when_value, then_expr in homog_case_item_list
            ],
            else_expr=else_expr,
        )

    def case_item_expr_heterogeneous(self, s):
        predicate, then_expr = s

        assert isinstance(then_expr, TransformedExpr), f"{then_expr} {type(then_expr)}"

        if isinstance(predicate, list):
            # If we have multible predicates for same 'then_expr'
            preds = []
            then_exprs = []
            for pred_i in predicate:
                preds.append(pred_i)
                then_exprs.append(then_expr)
            return preds, then_exprs

        else:
            assert isinstance(
                predicate, TransformedPredicate
            ), f"{predicate} {type(predicate)}"

        return predicate, then_expr

    def case_expr_heterogenous(self, s):
        case_item_list, else_expr = s[:-1], s[-1]
        return self._case_item_to_expr(case_item_list, else_expr)

    def _format_case_item(
        self, case_item: Tuple[List[TransformedPredicate], List[TransformedExpr]]
    ) -> List[Tuple[TransformedPredicate, TransformedExpr]]:

        assert len(case_item[0]) == len(case_item[1])
        case_item_formatted = [
            (pred, expr) for pred, expr in zip(case_item[0], case_item[1])
        ]
        return case_item_formatted

    def _format_case_item_list(
        self, case_item_list: List[Tuple[TransformedPredicate, TransformedExpr]]
    ):

        assert isinstance(case_item_list, list)
        case_item_list_formatted = []
        for case_item in case_item_list:
            assert isinstance(case_item, tuple)
            if isinstance(case_item[0], list) and isinstance(case_item[1], list):
                case_item_list_formatted.extend(self._format_case_item(case_item))
            else:
                case_item_list_formatted.append(case_item)

        return case_item_list_formatted

    def _expand_case_item_list(
        self, case_item_list: List[Tuple[TransformedPredicate, TransformedExpr]]
    ):
        """In case we have something like
            WHEN A THEN (WHEN B THEN 1 ELSE 2) ELSE 3
        we want to transform it into
            WHEN (A AND B) THEN 1 WHEN (A AND NOT B THEN 2) ELSE 3
        """

        out = []

        for predicate, then_expr in case_item_list:

            if then_expr.pred_list:
                for pred_i, value_i in then_expr.to_list():
                    if isinstance(predicate, TransformedPredicate):
                        new_pred_i = TransformedPredicate(
                            pred=And([predicate.pred] + [pred_i])
                        )
                        new_value_i = TransformedExpr.from_value(value_i)
                        out.append((new_pred_i, new_value_i))
                    else:
                        raise NotImplementedError(f"type(pred) = {type(predicate)}")
            else:
                out.append((predicate, then_expr))

        return out

    def _case_item_to_expr(self, case_item_list: List[Tuple[TransformedPredicate, TransformedExpr]],
                           else_expr: TransformedExpr) -> TransformedExpr:
        case_item_list = self._format_case_item_list(case_item_list)

        # Compute else
        assert len(case_item_list) > 0
        else_pred = self._compute_else_pred(case_item_list)

        # Add else to case_item_list
        case_item_list.append((TransformedPredicate(else_pred), else_expr))

        # Expand
        case_item_list_expanded = self._expand_case_item_list(case_item_list)

        pred_list = [predicate.to_dict() for predicate, _ in case_item_list_expanded]
        value_list = [then_expr.to_value() for _, then_expr in case_item_list_expanded]

        return TransformedExpr(pred_list=pred_list, value_list=value_list)

    def _compute_else_pred(self, case_item_list) -> GenericRule:
        pred_list = [predicate.to_dict() for predicate, _ in case_item_list]
        if len(pred_list) == 1:
            else_pred: GenericRule = Not(pred_list[0])
        else:
            else_pred = And([Not(pred) for pred in pred_list])

        return else_pred

    def case_expr(self, s):
        return s[0]

    def expr(self, s):
        return s[0]

    def start(self, s):
        (s,) = s
        if isinstance(s, TransformedExpr):
            return s.to_list()
        elif isinstance(s, Tree):
            return s.pretty()
        elif type(s) in [str, float, int, type(None)]:
            return TransformedExpr.from_value(s).to_list()
        else:
            raise ValueError(s)


def get_null_cols(rule: Union[List, Tuple, GenericRule]) -> Tuple[List[str], List[str]]:
    """Given a rule, find all null and not null columns, and return (list_null_cols, list_not_null_cols)"""

    if isinstance(rule, (tuple, list)):
        return _get_null_cols_iterable(rule)

    elif isinstance(rule, IsNull) and isinstance(rule.argument, Column):
        return ([rule.argument.column_name], [])

    elif isinstance(rule, Not):
        return _get_null_cols_not(rule)

    if isinstance(rule, Equals):
        return _get_null_cols_equals(rule)

    elif isinstance(rule, IsIn):
        return _get_null_cols_isin(rule)

    elif isinstance(rule, LogicComparator):
        return _get_null_cols_iterable(rule.arguments)

    return [], []


def _get_null_cols_iterable(iterable: Union[List, Tuple]) -> Tuple[List[str], List[str]]:
    null_cols = []
    not_null_cols = []
    for r in iterable:
        null_cols_i, not_null_cols_i = get_null_cols(r)
        null_cols.extend(null_cols_i)
        not_null_cols.extend(not_null_cols_i)
    return null_cols, not_null_cols


def _get_null_cols_not(rule: Not) -> Tuple[List[str], List[str]]:
    arg = rule.argument
    if isinstance(arg, IsNull) or isinstance(arg, LogicComparator):
        not_null_cols, null_cols = get_null_cols(arg)
        return null_cols, not_null_cols
    elif isinstance(arg, Equals):
        return _get_null_cols_equals(arg, inverse=True)
    elif isinstance(arg, IsIn):
        return _get_null_cols_isin(arg, inverse=True)

    return [], []


def _get_null_cols_equals(rule: Equals, inverse: bool = False) -> Tuple[List[str], List[str]]:
    if not (isinstance(rule.v1, Column) and isinstance(rule.v2, Value)):
        return [], []

    if not inverse and rule.v2.value is None:
        return [rule.v1.column_name], []
    elif not inverse and rule.v2.value is not None:
        return [], [rule.v1.column_name]
    elif inverse and rule.v2.value is None:
        return [], [rule.v1.column_name]
    return [], []


def _get_null_cols_isin(rule: IsIn, inverse: bool = False) -> Tuple[List[str], List[str]]:
    if not (isinstance(rule.v1, Column) and all([isinstance(v2i, Value) for v2i in rule.v2])):
        return [], []

    if not inverse and all([cast(Value, v2i).value is None for v2i in rule.v2]):
        return [rule.v1.column_name], []
    elif not inverse and all([cast(Value, v2i).value is not None for v2i in rule.v2]):
        return [], [rule.v1.column_name]
    elif inverse and all([cast(Value, v2i).value is None for v2i in rule.v2]):
        return [], [rule.v1.column_name]
    return [], []


def get_all_cols(rule: GenericRule) -> List[str]:
    """Get the list of all columns present in a rule."""
    return list(set([
        node.column_name for node in rule.get_children() if isinstance(node, Column)
    ]))


def is_consistent_rule(rule):
    """Check if a rule contains an inconsistent reference to a null and not null column.

    For example:
    >>> rule = And([IsNull(TableColumn("A")), Not(IsNull(TableColumn("A"))])
    >>> is_consistent_rule(rule)
    >>> False

    >>> rule = And([IsNull(TableColumn("A")), Not(IsNull(TableColumn("B"))])
    >>> is_consistent_rule(rule)
    >>> True
    """

    null_cols, not_null_cols = get_null_cols(rule)

    if len(set(null_cols) & set(not_null_cols)):
        return False
    return True


def uniformed_program(program: str) -> str:
    program = program.replace("\n", " ")
    program = program.replace("_X000D_", "")
    return program


def parse_rule_stmt(stmt: str, column_name_list: Optional[List[str]] = None, check_duplicated_rules: bool = True,
                    new_rules: Optional[Dict[str, Callable]] = None) -> ParsedRuleStmt:
    """
    Transforms the functional mapping statement into the list of logical branches.

    Example of functional mapping:
    ```
    CASE
        WHEN KRO.STATUS = 'FINISHED' THEN 'OK'
        WHEN KRO.STATUS = 'FAILED' THEN 'FAILED'
        ELSE 'IN_PROGRESS'
    END
    ```
    This functional mapping has three logical branches:
        - if STATUS = 'FINISHED' => 'OK'
        - if STATUS = 'FAILED' => 'FAILED'
        - else => 'IN_PROGRESS'

    Each branch is specified by pair of (logical predicate, value). This function
        parses the initial statement into the list of those pairs.

    The ParsedRuleStmt result looks like this:
    ```
    [
        (Equals(TableColumn('KRO.STATUS'), Value('FINISHED')), Value('OK')),
        (Equals(TableColumn('KRO.STATUS'), Value('FAILED')), Value('FAILED')),
        (And([
                Not(Equals(TableColumn('KRO.STATUS'), Value('FINISHED'))),
                Not(Equals(TableColumn('KRO.STATUS'), Value('FAILED')))
            ]), Value('IN_PROGRESS'))
    ]
    ```

    See unit tests for additional examples.
    """

    uniform_stmt = uniformed_program(stmt)
    parsed_tree = rule_lr.parse(uniform_stmt)
    # print(parsed_tree.pretty())
    return parse_rule_stmt_from_tree(parsed_tree=parsed_tree, column_name_list=column_name_list,
                                     check_duplicated_rules=check_duplicated_rules, new_rules=new_rules)


def parse_rule_stmt_from_tree(parsed_tree: Tree, column_name_list: Optional[List[str]] = None,
                              check_duplicated_rules: bool = True,
                              new_rules: Optional[Dict[str, Callable]] = None) -> ParsedRuleStmt:

    new_rules_dict: Dict[str, Callable] = new_rules if new_rules else {}

    tree_transformer = RuleTransformer(
        columns=column_name_list,
        new_rules=new_rules_dict,
    )

    result_list = tree_transformer.transform(parsed_tree)
    if check_duplicated_rules:
        result_list = list(filter(is_consistent_rule, result_list))

    return cast(ParsedRuleStmt, list(result_list))
