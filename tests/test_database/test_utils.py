from unittest.mock import MagicMock

import pytest
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from insight.database.utils import get_version_id


@pytest.fixture
def mock_session():
    session = MagicMock()
    session.begin_nested.return_value.__enter__.return_value = session

    def add_side_effect(model_instance):
        model_instance.id = 123

    session.add.side_effect = add_side_effect
    session.commit = MagicMock()

    session.rollback = MagicMock()

    mock_scalar_one_or_none = MagicMock()
    mock_scalar_one_or_none.id = 789
    executed = MagicMock()
    executed.scalar_one_or_none.return_value = mock_scalar_one_or_none
    session.execute.return_value = executed

    assert session.execute().scalar_one_or_none().id == 789
    return session


def test_get_version_id_existing_version(mock_session):
    mock_session.execute.return_value.scalar_one_or_none.return_value = MagicMock(id=123)
    version_id = get_version_id("existing_version", mock_session)
    assert version_id == 123


def test_get_version_id_new_version(mock_session):
    mock_session.execute.return_value.scalar_one_or_none.return_value = None
    mock_session.begin_nested.return_value.__enter__.return_value.add.return_value = MagicMock(
        id=123
    )
    version_id = get_version_id("new_version", mock_session)
    assert version_id == 123


def test_get_version_id_integrity_error(mock_session):
    # First call to execute raises IntegrityError
    # Second call to execute returns a MagicMock with the correct id
    second_execute = MagicMock()
    second_execute.scalar_one_or_none.return_value = MagicMock(id=789)
    mock_session.execute.side_effect = [
        IntegrityError("Mocked Integrity Error", "params", "orig"),
        second_execute,
    ]

    version_id = get_version_id("version_with_error", mock_session)

    assert version_id == 789


def test_get_version_id_sqlalchemy_error(mock_session):
    mock_session.execute.side_effect = SQLAlchemyError("Mocked SQLAlchemy Error")
    with pytest.raises(SQLAlchemyError):
        get_version_id("version_with_sqlalchemy_error", mock_session)
