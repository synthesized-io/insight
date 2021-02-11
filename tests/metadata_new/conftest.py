import pytest

pytest_plugins = [
   "tests.metadata_new.dataframes",
]


@pytest.fixture(scope='package')
def name():
    """A default name used when testing metas."""
    return 'meta'
