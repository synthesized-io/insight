import os
from typing import Optional

from lark import Lark


def load_grammar(
    start: str,
    grammar_path: str = "grammar.lark",
    location: Optional[str] = None,
    parser: str = "lalr",
) -> Lark:
    if location is None:
        location = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__))
        )
    with open(os.path.join(location, grammar_path)) as grammar_fd:
        lr = Lark(f"start: {start}\n" + grammar_fd.read(), parser=parser)  # type: ignore
    return lr
