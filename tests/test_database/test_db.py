from datetime import date, datetime
from math import isclose

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from insight import metrics
from insight.database import utils
from insight.database.schema import Base


@pytest.fixture(scope="module")
def db_session():
    """
    Sets up and destroyes an in-memory database for testing of database uploads.

    Returns: Session object which can be used to connect to the database.
    """
    # Setup
    engine = create_engine("sqlite:///:memory:")
    Session = sessionmaker(bind=engine, future=True)
    Base.metadata.create_all(engine)
    yield Session

    # Teardown
    Base.metadata.drop_all(engine)


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
    df.name = "test_dataset"
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

    if num_rows is None:
        assert dataset_table[2] is None
    else:
        assert dataset_table[2] == num_rows
    if num_cols is None:
        assert dataset_table[3] is None
    else:
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


def test_one_column_metric_queries_default_params(db_session, dataset, column):
    metric = metrics.Mean()
    with db_session(expire_on_commit=False) as session:
        metric(column, session)
        verify_results_table(session, metric(column))
        verify_dataset_table(session, "Series_age", None, 1)
        verify_metric_table(session, 'mean', 'OneColumnMetric')
        verify_version_table(session, "v1.10")


def test_one_column_metric_queries_modified_params(db_session, dataset, column):
    metric = metrics.Mean()
    with db_session(expire_on_commit=False) as session:
        metric(column, session, num_rows=5, dataset_name="testing_name", version="v2.0")
        verify_dataset_table(session, "testing_name", 5, 1)
        verify_version_table(session, "v2.0")


def test_two_column_metric_queries_default_params(db_session, dataset, column):
    metric = metrics.Norm()
    with db_session(expire_on_commit=False) as session:
        metric(column, column, session)
        verify_results_table(session, metric(column, column))
        verify_dataset_table(session, "Series_age", None, 1)
        verify_metric_table(session, "norm", "TwoColumnMetric")
        verify_version_table(session, "v1.10")


def test_two_column_metric_queries_modified_params(db_session, dataset, column):
    metric = metrics.Norm()
    with db_session(expire_on_commit=False) as session:
        metric(column, column, session, num_rows=5, dataset_name="testing_name", version="v2.0")
        verify_dataset_table(session, "testing_name", 5, 1)
        verify_version_table(session, "v2.0")


def test_dataframe_queries_default_params(db_session, dataset):
    metric = metrics.OneColumnMap(metrics.Mean())
    with db_session(expire_on_commit=False) as session:
        metric(dataset, session=session, dataset_name="test_dataset")
        verify_results_table(session, metric.summarize_result(metric(dataset)))
        verify_dataset_table(session, "test_dataset", None, None)
        verify_metric_table(session, "mean_map", "DataFrameMetric")
        verify_version_table(session, "v1.10")


def test_dataframe_queries_modified_params(db_session, dataset):
    metric = metrics.OneColumnMap(metrics.Mean())
    with db_session(expire_on_commit=False) as session:
        metric(dataset, session=session, dataset_name="test_dataset", dataset_rows=5, dataset_cols=6, version="v2.0")
        verify_dataset_table(session, "test_dataset", num_rows=5, num_cols=6)
        verify_version_table(session, "v2.0")


def test_two_dataframe_queries_default_params(db_session, dataset):
    metric = metrics.TwoColumnMap(metrics.KullbackLeiblerDivergence())
    with db_session(expire_on_commit=False) as session:
        metric(dataset, dataset, session=session, dataset_name="test_dataset")
        verify_results_table(session, metric.summarize_result(metric(dataset, dataset)))
        verify_dataset_table(session, "test_dataset", None, None)
        verify_metric_table(session, "kullback_leibler_divergence_map", "TwoDataFrameMetrics")
        verify_version_table(session, "v1.10")


def test_two_dataframe_queries_modified_params(db_session, dataset):
    metric = metrics.TwoColumnMap(metrics.KullbackLeiblerDivergence())
    with db_session(expire_on_commit=False) as session:
        metric(dataset, dataset, session=session,
               dataset_name="test_dataset", dataset_rows=5, dataset_cols=6, version="v2.0")
        verify_dataset_table(session, "test_dataset", num_rows=5, num_cols=6)
        verify_version_table(session, version="v2.0")
