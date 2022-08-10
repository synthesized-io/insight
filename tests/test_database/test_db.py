from datetime import date, datetime
from math import isclose

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from insight import metrics
from insight.database import utils
from insight.database.schema import Base


@pytest.fixture(scope="session")
def engine():
    return create_engine("sqlite:///:memory:")


@pytest.fixture(scope="session")
def tables(engine):
    Base.metadata.create_all(engine)
    yield
    Base.metadata.drop_all(engine)


@pytest.fixture
def db_session(engine, tables):
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, expire_on_commit=False)

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture(scope="module")
def dataset():
    """
    Retrieves a dataset for testing from github.

    Returns: a pandas dataframe
    """
    data_directory = "https://raw.githubusercontent.com/synthesized-io/datasets/master"
    dataset = "tabular/templates/credit.csv"
    url = f"{data_directory}/{dataset}"

    df = utils.get_df(url)
    df = df.dropna()

    return df


@pytest.fixture(scope="module")
def column(dataset):
    """
    Gives a column from the dataset that is used for testing.
    """
    return dataset['age']


def verify_results_table(session, result):
    """
    Verifies the results table in the database.
    """
    results_table = session.execute(text("SELECT * FROM Result")).fetchall()[0]
    assert results_table[0] == results_table[1] == results_table[2] == results_table[3] == 1
    assert isclose(results_table[4], result)
    assert datetime.strptime(results_table[5], '%Y-%m-%d %H:%M:%S').date() == date.today()


def verify_dataset_table(session, table_name, num_rows, num_cols):
    """
    Verifies the dataset table in the database.
    """
    dataset_table = session.execute(text("SELECT * FROM dataset WHERE id == 1")).fetchall()[0]
    assert dataset_table[0] == 1
    assert dataset_table[1] == table_name
    assert dataset_table[2] == num_rows
    assert dataset_table[3] == num_cols


def verify_metric_table(session, name, category):
    """
    Verifies the metrics table in the database.
    """
    metric_table = session.execute(text("SELECT * FROM metric where id == 1")).fetchall()[0]
    assert metric_table[0] == 1
    assert metric_table[1] == name
    assert metric_table[2] == category


def verify_version_table(session, version):
    """
    Verifies the version table in the database.
    """
    version_table = session.execute(text("SELECT * FROM version where id == 1")).fetchall()[0]
    assert version_table[0] == 1
    assert version_table[1] == version


def test_one_column_metric_queries_default_params(db_session, column):
    metric = metrics.Mean()
    metric(column, db_session)
    verify_results_table(db_session, metric(column))
    verify_dataset_table(db_session, "Series_age", len(column), 1)
    verify_metric_table(db_session, 'mean', 'OneColumnMetric')
    verify_version_table(db_session, "Unversioned")


def test_one_column_metric_queries_modified_params(db_session, column):
    metric = metrics.Mean()
    metric(column, db_session, dataset_name="testing_name")
    verify_dataset_table(db_session, "testing_name", len(column), 1)


def test_two_column_metric_queries_default_params(db_session, column):
    metric = metrics.Norm()
    metric(column, column, db_session)
    verify_results_table(db_session, metric(column, column))
    verify_dataset_table(db_session, "Series_age", len(column), 1)
    verify_metric_table(db_session, "norm", "TwoColumnMetric")
    verify_version_table(db_session, "Unversioned")


def test_two_column_metric_queries_modified_params(db_session, column):
    metric = metrics.Norm()
    metric(column, column, db_session, dataset_name="testing_name")
    verify_dataset_table(db_session, "testing_name", len(column), 1)


def test_dataframe_queries_default_params(db_session, dataset):
    metric = metrics.OneColumnMap(metrics.Mean())
    metric(dataset, session=db_session, dataset_name="test_dataset")
    verify_results_table(db_session, metric.summarize_result(metric(dataset)))
    verify_dataset_table(db_session, "test_dataset", num_rows=dataset.shape[0], num_cols=dataset.shape[1])
    verify_metric_table(db_session, "mean_map", "DataFrameMetric")
    verify_version_table(db_session, "Unversioned")


def test_dataframe_queries_modified_params(db_session, dataset):
    metric = metrics.OneColumnMap(metrics.Mean())
    metric(dataset, session=db_session, dataset_name="test_dataset")
    verify_dataset_table(db_session, "test_dataset", num_rows=dataset.shape[0], num_cols=dataset.shape[1])


def test_two_dataframe_queries_default_params(db_session, dataset):
    metric = metrics.TwoColumnMap(metrics.KullbackLeiblerDivergence())
    metric(dataset, dataset, session=db_session, dataset_name="test_dataset")
    verify_results_table(db_session, metric.summarize_result(metric(dataset, dataset)))
    verify_dataset_table(db_session, "test_dataset", num_rows=dataset.shape[0], num_cols=dataset.shape[1])
    verify_metric_table(db_session, "kullback_leibler_divergence_map", "TwoDataFrameMetrics")
    verify_version_table(db_session, "Unversioned")


def test_two_dataframe_queries_modified_params(db_session, dataset):
    metric = metrics.TwoColumnMap(metrics.KullbackLeiblerDivergence())
    metric(dataset, dataset, session=db_session,
           dataset_name="test_dataset")
    verify_dataset_table(db_session, "test_dataset", num_rows=dataset.shape[0], num_cols=dataset.shape[1])
