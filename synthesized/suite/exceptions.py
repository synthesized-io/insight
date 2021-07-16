from lark.exceptions import LarkError


class SuiteReadError(Exception):
    pass


class ParserError(LarkError):
    pass
