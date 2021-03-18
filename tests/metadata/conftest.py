import pytest


@pytest.fixture(scope='package')
def name():
    """A default name used when testing metas."""
    return 'meta'


@pytest.fixture(scope='class', params=[False, True], ids=['complete', 'with nans'])
def with_nans(request):
    return request.param
