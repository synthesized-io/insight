import pytest

pytest_plugins = [
   "tests.metadata_new.dataframes",
]


@pytest.fixture(scope='package')
def name():
    """A default name used when testiing metas."""
    return 'meta'
