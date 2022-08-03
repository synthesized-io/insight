"""Utils for fetching information from the backend DB."""
import re
import typing

import pandas as pd
from sqlalchemy.sql import select

import insight.database.schema as model
import insight.database.db_connection as connection

ModelType = typing.TypeVar('ModelType', bound='model.Base')


def get_df_and_id(url_or_path: str):
    matched = re.match(r".*\/([a-zA-Z0-9\-_]+)(\.\w{1,4})?", url_or_path)
    if matched is None:
        raise ValueError()
    df = pd.read_csv(url_or_path)
    name = matched.group(1)
    dataset = get_object_from_name(name, model.Dataset)
    return df, dataset.id


def get_metric_id(metric):
    db_metric = get_object_from_name(metric.name, model.Metric)
    return db_metric.id


def get_version_id(version: str):
    db_version = get_object_from_name(version, model.Version)
    return db_version.id


def get_object_from_name(name: str, model_cls: typing.Type[ModelType]) -> ModelType:
    with connection.Session(expire_on_commit=False) as session:
        result = session.execute(
            select(model_cls).where(model_cls.name == name)
        ).scalar_one_or_none()

        if result is not None:
            obj = result
        else:
            obj = model_cls(name=name)
            session.add(obj)
            session.commit()
    return obj
