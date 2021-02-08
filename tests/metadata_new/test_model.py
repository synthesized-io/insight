import logging
from datetime import datetime
from typing import cast

import numpy as np
import pandas as pd
import pytest

from synthesized.metadata_new import Date, Integer, MetaExtractor
from synthesized.metadata_new.model import FormattedString, Histogram, KernelDensityEstimate, ModelFactory

logger = logging.getLogger(__name__)


@pytest.fixture
def simple_df():
    np.random.seed(6235901)
    df = pd.DataFrame({
        'string': np.random.choice(['A','B','C','D','E'], size=1000),
        'bool': np.random.choice([False, True], size=1000).astype('?'),
        'date': pd.to_datetime(18_000 + np.random.normal(500, 50, size=1000).astype(int), unit='D'),
        'int': [n for n in [0, 1, 2, 3, 4, 5] for i in range([50, 50, 0, 200, 400, 300][n])],
        'float': np.random.normal(0.0, 1.0, size=1000),
        'int_bool': np.random.choice([0, 1], size=1000),
        'date_sparse': pd.to_datetime(18_000 + 5 * np.random.normal(500, 50, size=1000).astype(int), unit='D')
    })
    return df


@pytest.fixture
def simple_df_binned_probabilities():
    probs = {
        'date': {pd.Interval(pd.Timestamp(datetime.strptime('2020-02-24 00:00:00', "%Y-%m-%d %H:%M:%S")), pd.Timestamp(datetime.strptime('2020-03-30 07:12:00', "%Y-%m-%d %H:%M:%S")), closed='left'): 0.0032244955008293133, pd.Interval(pd.Timestamp(datetime.strptime('2020-03-30 07:12:00', "%Y-%m-%d %H:%M:%S")), pd.Timestamp(datetime.strptime('2020-05-04 14:24:00', "%Y-%m-%d %H:%M:%S")), closed='left'): 0.011699174133465277, pd.Interval(pd.Timestamp(datetime.strptime('2020-05-04 14:24:00', "%Y-%m-%d %H:%M:%S")), pd.Timestamp(datetime.strptime('2020-06-08 21:36:00', "%Y-%m-%d %H:%M:%S")), closed='left'): 0.05498513716790392, pd.Interval(pd.Timestamp(datetime.strptime('2020-06-08 21:36:00', "%Y-%m-%d %H:%M:%S")), pd.Timestamp(datetime.strptime('2020-07-14 04:48:00', "%Y-%m-%d %H:%M:%S")), closed='left'): 0.14416831367994937, pd.Interval(pd.Timestamp(datetime.strptime('2020-07-14 04:48:00', "%Y-%m-%d %H:%M:%S")), pd.Timestamp(datetime.strptime('2020-08-18 12:00:00', "%Y-%m-%d %H:%M:%S")), closed='left'): 0.22353093217772335, pd.Interval(pd.Timestamp(datetime.strptime('2020-08-18 12:00:00', "%Y-%m-%d %H:%M:%S")), pd.Timestamp(datetime.strptime('2020-09-22 19:12:00', "%Y-%m-%d %H:%M:%S")), closed='left'): 0.24885100781537467, pd.Interval(pd.Timestamp(datetime.strptime('2020-09-22 19:12:00', "%Y-%m-%d %H:%M:%S")), pd.Timestamp(datetime.strptime('2020-10-28 02:24:00', "%Y-%m-%d %H:%M:%S")), closed='left'): 0.18608688433096188, pd.Interval(pd.Timestamp(datetime.strptime('2020-10-28 02:24:00', "%Y-%m-%d %H:%M:%S")), pd.Timestamp(datetime.strptime('2020-12-02 09:36:00', "%Y-%m-%d %H:%M:%S")), closed='left'): 0.09085677663066802, pd.Interval(pd.Timestamp(datetime.strptime('2020-12-02 09:36:00', "%Y-%m-%d %H:%M:%S")), pd.Timestamp(datetime.strptime('2021-01-06 16:48:00', "%Y-%m-%d %H:%M:%S")), closed='left'): 0.03138891193009287, pd.Interval(pd.Timestamp(datetime.strptime('2021-01-06 16:48:00', "%Y-%m-%d %H:%M:%S")), pd.Timestamp(datetime.strptime('2021-02-11 00:00:00', "%Y-%m-%d %H:%M:%S")), closed='left'): 0.005208366633031028},
        'int': {pd.Interval(0, 1, closed='left'): 0.060437068579862134, pd.Interval(1, 2, closed='left'): 0.030799447835097436, pd.Interval(2, 3, closed='left'): 0.12191977892474855, pd.Interval(3, 4, closed='left'): 0.3633195010374817, pd.Interval(4, 5, closed='left'): 0.42352420362280985},
        'float': {pd.Interval(-3.6625330162590917, -2.930615228589776, closed='left'): 0.002515169527140779, pd.Interval(-2.930615228589776, -2.1986974409204603, closed='left'): 0.016909613332347164, pd.Interval(-2.1986974409204603, -1.4667796532511446, closed='left'): 0.05968321861823208, pd.Interval(-1.4667796532511446, -0.7348618655818289, closed='left'): 0.1505943457380943, pd.Interval(-0.7348618655818289, -0.0029440779125131655, closed='left'): 0.26938996058883313, pd.Interval(-0.0029440779125131655, 0.7289737097568025, closed='left'): 0.25887642078328754, pd.Interval(0.7289737097568025, 1.4608914974261182, closed='left'): 0.16429211636779523, pd.Interval(1.4608914974261182, 2.192809285095434, closed='left'): 0.06263625372938374, pd.Interval(2.192809285095434, 2.9247270727647496, closed='left'): 0.012323832964755947, pd.Interval(2.9247270727647496, 3.6566448604340653, closed='left'): 0.00277906835012976},
        'int_bool': {pd.Interval(0, 1, closed='left'): 1.0}
    }
    return probs


@pytest.fixture
def simple_df_meta(simple_df):
    df_meta = MetaExtractor.extract(simple_df)
    return df_meta


@pytest.mark.slow
@pytest.mark.parametrize("col", ['string',  'bool', 'date', 'int', 'float', 'int_bool'])
def test_histogram_from_meta(col, simple_df, simple_df_meta):
    """Test basic construction of histograms."""
    hist = Histogram.from_meta(simple_df_meta[col])
    logger.info(hist)
    hist.fit(simple_df)
    hist.plot()


@pytest.mark.slow
@pytest.mark.parametrize("col", ['date', 'int', 'float', 'int_bool'])
def test_histogram_from_binned_affine(col, simple_df, simple_df_meta):
    """Test construction of histograms from binning affine values."""
    hist = Histogram.bin_affine_meta(simple_df_meta[col], max_bins=20)
    logger.info("%s -> %s", simple_df_meta[col], hist)
    logger.info("Num Bins: %d", len(hist.categories))
    assert len(hist.categories) <= 20
    hist.fit(simple_df)
    hist.plot()


@pytest.mark.fast
def test_histogram_from_affine_precision_int(simple_df, simple_df_meta):
    """For Integers, If the Histogram comes from a meta with a precision that spans multiple values, it should bin the
    entire range using the defined precision. Otherwise, it should just return the specific values.
    """
    col = "int"
    int_meta = cast(Integer, simple_df_meta[col])

    logger.debug(int_meta.categories)  # [0, 1, 3, 4, 5]
    logger.debug("precision: %s", int_meta.unit_meta.precision)  # 1

    hist = Histogram.from_meta(int_meta)
    logger.debug(hist)
    logger.debug(hist.categories)  # [0, 1, 3, 4, 5]
    assert hist.dtype == "i8"
    assert hist.categories == int_meta.categories

    # Now we increase the precision to span multiple values.
    int_meta.unit_meta.precision = np.int64(2)
    hist = Histogram.from_meta(int_meta)
    logger.debug(hist)
    logger.debug(hist.categories)  # [[0, 2), [2, 4), [4, 6)]

    assert hist.dtype in ["interval[int64]", "interval[i8]"]
    assert len(hist.categories) == 3


@pytest.mark.fast
def test_histogram_from_affine_precision_date(simple_df, simple_df_meta):
    """For Date values, If the Histogram comes from a meta with a precision that spans multiple values, it should bin
    the entire range using the defined precision. Otherwise, it should just return the specific values.
    """
    col = "date_sparse"
    date_meta = cast(Date, simple_df_meta[col])

    logger.debug(date_meta.categories[:3])  # [numpy.datetime64('2023-07-07'), numpy.datetime64('2023-10-15'), ...]

    logger.debug("precision: %s", date_meta.unit_meta.precision)  # np.timedelta64(1, 'D')

    hist = Histogram.from_meta(date_meta)
    assert hist.dtype == "M8[D]"
    assert hist.categories == date_meta.categories

    # Now we increase the precision, but it doesn't span multiple values yet. (smallest diff is 5 days)
    date_meta.unit_meta.precision = np.timedelta64(3, 'D')
    hist = Histogram.from_meta(date_meta)
    assert hist.dtype == "M8[D]"
    assert hist.categories == date_meta.categories

    # Finally we increase the precision so that it spans multiple values
    date_meta.unit_meta.precision = np.timedelta64(10, 'D')
    hist = Histogram.from_meta(date_meta)
    logger.debug(hist)
    logger.debug(hist.categories[:3])  # [[2023-07-07, 2023-07-17), [2023-07-17, 2023-07-27), [2023-07-27, 2023-08-06)]

    assert hist.dtype in ["interval[datetime64[ns]]", "interval[M8[ns]]"]
    assert len(hist.categories) == 181


@pytest.mark.slow
@pytest.mark.parametrize("col", ['date', 'int', 'float', 'int_bool'])
def test_kde_model(col, simple_df_binned_probabilities, simple_df, simple_df_meta):
    kde = KernelDensityEstimate.from_meta(simple_df_meta[col])
    logger.info(kde)
    kde.fit(simple_df)
    kde.plot()
    hist = Histogram.bin_affine_meta(kde, max_bins=10)
    assert hist.probabilities == simple_df_binned_probabilities[col]


@pytest.mark.fast
def test_formatted_string_model():
    pattern = '[0-9]{5}'
    model = FormattedString('test', [pattern], nan_freq=0.3)
    assert model.sample(100)['test'].str.match(pattern).sum() == 100
    assert model.sample(100, produce_nans=True)['test'].isna().sum() > 0

    patterns = ['[0-9]{5}', 'AB[0-9]{3}']
    model = FormattedString('test', patterns, probabilities={'[0-9]{5}': 0.3, 'AB[0-9]{3}': 0.7}, nan_freq=0.3)
    assert model.sample(100)['test'].str.match('[0-9]{5}|AB[0-9]{3}').sum() == 100
    assert model.sample(100, produce_nans=True)['test'].isna().sum() > 0


@pytest.mark.fast
def test_factory(simple_df_meta):
    df_models = ModelFactory().create_model(simple_df_meta)

    assert isinstance(ModelFactory().create_model(simple_df_meta), dict)
    assert isinstance(ModelFactory().create_model(simple_df_meta['bool']), (Histogram, KernelDensityEstimate))
    assert isinstance(df_models['string'], Histogram)
    assert isinstance(df_models['bool'], Histogram)
    assert isinstance(df_models['date'], KernelDensityEstimate)
    assert isinstance(df_models['int'], Histogram)
    assert isinstance(df_models['float'], KernelDensityEstimate)
    assert isinstance(df_models['int_bool'], Histogram)
