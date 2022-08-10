"""Utils for fetching information from the backend DB."""
import re
import typing

import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy.sql import select

import insight.database.schema as model

NamedModelType = typing.TypeVar('NamedModelType', model.Dataset, model.Metric, model.Version)


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

    return dataset.id


def get_metric_id(metric, session: Session, category: str = None):
    db_metric = get_object_from_db_by_name(metric.name, session, model.Metric)

    if db_metric is None:
        with session:
            db_metric = model.Metric(name=metric.name, category=category)
            session.add(db_metric)
            session.commit()

    return db_metric.id


def get_version_id(version: str, session: Session):
    db_version = get_object_from_db_by_name(version, session, model.Version)
    if db_version is None:
        with session:
            db_version = model.Version(name=version)
            session.add(db_version)
            session.commit()

    return db_version.id


def get_object_from_db_by_name(name: str,
                               session: Session,
                               model_cls: typing.Type[NamedModelType]) -> typing.Union[NamedModelType, None]:
    with session:
        result = session.execute(
            select(model_cls).where(model_cls.name == name)
        ).scalar_one_or_none()
        return result
