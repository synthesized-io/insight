MAJOR: int = 1
MINOR: int = 2
PATCH: int = 0
PRE_RELEASE: str = ''  # should start with "-" or be empty

__version__ = f'{MAJOR}.{MINOR}.{PATCH}{PRE_RELEASE}'
