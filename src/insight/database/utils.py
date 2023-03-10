"""Utils for fetching information from the backend DB."""
import os
import re
import typing

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.sql import select

import insight.database.schema as model

NamedModelType = typing.TypeVar('NamedModelType', model.Dataset, model.Metric, model.Version)

_database_fail_note = "Failure to communicate with the database"


def get_df(url_or_path: str):
    matched = re.match(r".*\/([a-zA-Z0-9\-_]+)(\.\w{1,4})?", url_or_path)
    if matched is None:
        raise ValueError()
    df = pd.read_csv(url_or_path)
    df.name = matched.group(1)
    return df


def get_df_id(
        df_name: str, session: Session, num_rows: int = None, num_columns: int = None
) -> int:
    """Get the id of a dataframe in the database. If it doesn't exist, create it.

    Args:
        df_name (str): The name of the dataframe.
        session (Session): The database session.
        num_rows (int): The number of rows in the dataframe. Optional.
        num_columns (int): The number of columns in the dataframe. Optional.

    """
    dataset = get_object_from_db_by_name(df_name, session, model.Dataset)
    if dataset is None:
        with session:
            dataset = model.Dataset(name=df_name, num_columns=num_columns, num_rows=num_rows)
            session.add(dataset)
            session.commit()
    if not dataset.id:
        raise ConnectionError(_database_fail_note)
    return int(dataset.id)


def get_metric_id(metric: str, session: Session, category: str = None) -> int:
    """Get the id of a metric in the database. If it doesn't exist, create it.

    Args:
        metric (str): The name of the metric.
        session (Session): The database session.
        category (str): The category of the metric. Optional.
    """
    db_metric = get_object_from_db_by_name(metric, session, model.Metric)

    if db_metric is None:
        with session:
            db_metric = model.Metric(name=metric, category=category)
            session.add(db_metric)
            session.commit()
    if not db_metric.id:
        raise ConnectionError(_database_fail_note)
    return int(db_metric.id)


def get_version_id(version: str, session: Session) -> int:
    """Get the id of a version in the database. If it doesn't exist, create it.

    Args:
        version (str): The name of the version.
        session (Session): The database session.
    """
    db_version = get_object_from_db_by_name(version, session, model.Version)
    if db_version is None:
        with session:
            db_version = model.Version(name=version)
            session.add(db_version)
            session.commit()
    if not db_version.id:
        raise ConnectionError(_database_fail_note)
    return int(db_version.id)


def get_object_from_db_by_name(
        name: str, session: Session, model_cls: typing.Type[NamedModelType]
) -> typing.Union[NamedModelType, None]:
    """Get an object from the database by name.

    Args:
        name (str): The name of the object.
        session (Session): The database session.
        model_cls (typing.Type[NamedModelType]): The class of the object.
    """
    with session:
        result = session.execute(
            select(model_cls).where(model_cls.name == name)
        ).scalar_one_or_none()
        return result


def get_session() -> typing.Optional[Session]:
    """
    If a database exists, returns a sessionmaker object. Else returns None.
    Returns: sessionmaker object that can be used to access the database.

    """
    try:
        POSTGRES_PASSWORD = os.environ["POSTGRES_PASSWORD"]
        POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
        POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "5432")
        POSTGRES_USER = os.environ.get("POSTGRES_USER", "postgres")
        POSTGRES_DATABASE = os.environ.get("POSTGRES_DATABASE", "postgres")
        db_url = "postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@" \
                 "{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DATABASE}".format(
            POSTGRES_HOST=POSTGRES_HOST,
            POSTGRES_PORT=POSTGRES_PORT,
            POSTGRES_USER=POSTGRES_USER,
            POSTGRES_PASSWORD=POSTGRES_PASSWORD,
            POSTGRES_DATABASE=POSTGRES_DATABASE
        )
        engine = create_engine(db_url, future=True)

        session_constructor = sessionmaker(bind=engine, future=True)
        session = session_constructor(expire_on_commit=False)
    except KeyError:
        session = None

    return session
