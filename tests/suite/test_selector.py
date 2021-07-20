import pytest

from synthesized.suite.parsers import parse_join_stmt
from synthesized.suite.selector import PandaSQLSelector

from .cases import case_synthetic

selector, _, join_stmt_df = case_synthetic()


@pytest.mark.parametrize(
    "join_stmt", [join_stmt for _, (_, join_stmt) in join_stmt_df.iterrows()]
)
def test_suite_selector_Selector_join_unjoin_loop_tpch(join_stmt):

    join = parse_join_stmt(join_stmt)

    joined_df = selector.join(stmt=join, force_lowercase=False)
    unjoined_data = selector.unjoin(df=joined_df, stmt=join)

    unjoined_selector = PandaSQLSelector()
    unjoined_selector.data = unjoined_data
    unjoined_joined_df = unjoined_selector.join(stmt=join, force_lowercase=False)

    assert joined_df.equals(unjoined_joined_df)
