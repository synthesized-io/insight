import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier

from synthesized.insight.metrics import (CategoricalLogisticR2, CramersV, EarthMoversDistance, KendellTauCorrelation,
                                         KolmogorovSmirnovDistance, Mean, SpearmanRhoCorrelation, StandardDeviation)
from synthesized.insight.metrics.modelling_metrics import (Accuracy, F1Score, Precision, PredictiveModellingComparison,
                                                           PredictiveModellingScore, Recall, classifier_scores)

logger = logging.getLogger(__name__)

mean = Mean()
std_dev = StandardDeviation()
cramers_v = CramersV()
emd = EarthMoversDistance()
ksd = KolmogorovSmirnovDistance()
categorical_logistic_correlation = CategoricalLogisticR2()
kendell_tau_correlation = KendellTauCorrelation()


def test_mean():
    sr_a = pd.Series([1, 2, 3, 4, 5], name='a')
    val_a = mean(sr=sr_a)
    assert val_a == 3

    sr_b = pd.Series(np.datetime64('2020-01-01') + np.arange(0, 3, step=1).astype('m8[D]'), name='b')
    val_b = mean(sr=sr_b)
    assert val_b == np.datetime64('2020-01-02')

    sr_c = pd.Series(['a', 'b', 'c', 'd'], name='c')
    val_c = mean(sr=sr_c)
    assert val_c is None


def test_standard_deviation():
    sr_a = pd.Series(np.random.normal(0, 1, 100), name='a')
    val_a = std_dev(sr=sr_a)
    assert val_a is not None

    sr_b = pd.Series(np.datetime64('2020-01-01') + np.arange(0, 20, step=1).astype('m8[D]'), name='b')
    val_b = std_dev(sr=sr_b)
    assert val_b is not None

    sr_c = pd.Series(['a', 'b', 'c', 'd'], name='c')
    val_c = std_dev(sr=sr_c)
    assert val_c is None


def test_spearmans_rho():
    sr_a = pd.Series(np.random.normal(0, 1, 100), name='a')
    sr_b = pd.Series(np.random.normal(0, 1, 5), name='b')
    sr_c = pd.Series(sr_b.values + np.random.normal(0, 0.8, 5), name='c')
    sr_d = pd.Series(['a', 'b', 'c', 'd'], name='d')

    spearman_rho = SpearmanRhoCorrelation(max_p_value=0.05)

    assert spearman_rho(sr_a, sr_a) is not None
    assert spearman_rho(sr_b, sr_c) is None
    assert spearman_rho(sr_c, sr_d) is None


def test_em_distance():
    sr_a = pd.Series(np.random.normal(0, 1, 100), name='a')
    sr_b = pd.Series(['a', 'b', 'c', 'd'], name='b')

    assert emd(sr_a, sr_a) is None
    assert emd(sr_b, sr_b) is not None


def test_ks_distance():
    sr_a = pd.Series(np.random.normal(0, 1, 100), name='a')
    sr_b = pd.Series(['a', 'b', 'c', 'd'], name='b')

    assert ksd(sr_a, sr_a) is not None
    assert ksd(sr_b, sr_b) is None


def test_kt_correlation():
    sr_a = pd.Series(np.random.normal(0, 1, 100), name='a')
    sr_b = pd.Series(np.random.normal(0, 1, 5), name='b')
    sr_c = pd.Series(sr_b.values + np.random.normal(0, 0.8, 5), name='c')
    sr_d = pd.Series(['a', 'b', 'c', 'd'], name='d')

    kt_corr = KendellTauCorrelation(max_p_value=0.05)

    assert kt_corr(sr_a, sr_a) is not None
    assert kt_corr(sr_b, sr_c) is None
    assert kt_corr(sr_c, sr_d) is None


def test_cramers_v():
    sr_a = pd.Series([1, 2, 3, 1, 2, 3, 1, 2, 3] * 100, name='a')
    sr_b = pd.Series([1, 2, 3, 2, 3, 1, 3, 1, 2] * 100, name='b')

    assert cramers_v(sr_a, sr_a) > 0.99  # This metric -> cramers_v for large N (makes it more robust to outliers)
    assert cramers_v(sr_a, sr_b) == 0.

    sr_c = pd.Series(np.random.normal(0, 1, 1000), name='c')
    assert cramers_v(sr_c, sr_c) is None


def test_categorical_logistic_correlation():
    sr_a = pd.Series(np.random.normal(0, 1, 100), name='x')
    sr_a_dates = pd.Series(pd.date_range('01/01/01', '01/12/01', name='x'))
    sr_b = pd.Series(np.random.choice([1, 0], 100), name='y')

    value = categorical_logistic_correlation(sr_a, sr_a)
    assert value is None  # continuous, continuous -> None

    value = categorical_logistic_correlation(sr_b, sr_b)
    assert value is None  # categorical, categorical -> None

    value = categorical_logistic_correlation(sr_a_dates, sr_b)
    assert value is None  # continuous date, categorical -> None

    value = categorical_logistic_correlation(sr_a, sr_b)
    assert value is not None  # continuous, categorical -> not None


def test_categorical_logistic_correlation_datetimes():
    sr_a = pd.Series([1, 2, 3, 4, 5], name='ints')
    sr_b = pd.to_datetime(
        pd.Series(['10/07/2020', '10/06/2020', '10/12/2020', '1/04/2021', '10/06/2018'], name='dates')
    )

    value = categorical_logistic_correlation(sr_a, sr_b)


def test_kt_correlation_string_numbers():
    sr_a = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], name="a")
    sr_b = pd.Series(['1.4', '2.1', '', '4.1', '3.9', '4.4', '5.1', '6.0', '7.5', '9', '11.4', '12.1', '', '14.1', '13.9'], name="b")

    # df['TotalCharges'].dtype is Object in this case, eg. "103.4" instead of 103.4
    kt1 = kendell_tau_correlation(sr_a=sr_a, sr_b=sr_b)

    sr_b = pd.to_numeric(sr_b, errors='coerce')
    kt2 = kendell_tau_correlation(sr_a=sr_a, sr_b=sr_b)

    assert abs(kt1 - kt2) < 0.01


def test_predictive_modelling():

    pms = PredictiveModellingScore(model='Linear', y_label='y', x_labels=['x'])
    pmc = PredictiveModellingComparison(model='Linear', y_label='y', x_labels=['x'])

    x = np.random.normal(0, 1, 100)
    y = 2 * x + np.random.normal(0, 0.1, 100)
    y2 = 2.2 * x + np.random.normal(0, 0.1, 100)
    df_old = pd.DataFrame({'x': x, 'y': y})
    df_new = pd.DataFrame({'x': x, 'y': y2})

    assert pms(df_old) is not None
    assert pmc(df_old, df_new) is not None


# y_true has class 3, but y_pred doesn't
def test_classifier_scores_multiclass_classnotpresent_ypred():
    train = pd.DataFrame({
        'x1': ['100','100','100','200','200','300'],
        'x2': [1,2,3,1,4,5],
        'x3': [4,8,12,3,12,9],
        'y':  [1,1,1,2,2,3]
    })

    test = pd.DataFrame({
        'x1': ['100','200','200'],
        'x2': [4,2,5],
        'x3': [16,6,15],
        'y':  [1,2,3]
    })

    x_train = train[['x1','x2','x3']]
    y_train = train['y']
    x_test = test[['x1','x2','x3']]
    y_test = test['y']

    clf = RidgeClassifier()
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    assert((y_pred==np.array([1,2,2])).all() == True)
    assert((np.array(y_test)==np.array([1,2,3])).all() == True)

    metrics_dict = classifier_scores(x_train=np.array(x_train), y_train=np.array(y_train), \
                  x_test=np.array(x_test), y_test=np.array(y_test), clf = clf)

    assert(np.isclose(metrics_dict['roc_auc'],0.75,rtol=0.05) == True)


# y_true and y_pred both don't have class 4
def test_classifier_scores_multiclass_classnotpresent_both():
    train = pd.DataFrame({
        'x1': ['100','100','100','200','200','300','500'],
        'x2': [1,2,3,1,4,5,11],
        'x3': [4,8,12,3,12,9,4],
        'y':  [1,1,1,2,2,3,4]
    })

    test = pd.DataFrame({
        'x1': ['100','200','200'],
        'x2': [4,2,5],
        'x3': [16,6,15],
        'y':  [2,3,1]
    })

    x_train = train[['x1','x2','x3']]
    y_train = train['y']
    x_test = test[['x1','x2','x3']]
    y_test = test['y']

    clf = RidgeClassifier()
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    assert((y_pred==np.array([1,2,3])).all() == True)
    assert((np.array(y_test)==np.array([2,3,1])).all() == True)

    metrics_dict = classifier_scores(x_train=np.array(x_train), y_train=np.array(y_train), \
                  x_test=np.array(x_test), y_test=np.array(y_test), clf = clf)

    assert(np.isclose(metrics_dict['roc_auc'],0.25,rtol=0.05) == True)


# y_true doesn't have the class 3 but y_pred does
def test_classifier_scores_multiclass_classnotpresent_ytrue():
    train = pd.DataFrame({
        'x1': [100,100,100,200,200,300,500],
        'x2': [1,2,3,1,4,5,11],
        'x3': [4,8,12,3,12,9,22],
        'y':  [1,1,1,2,2,3,4]
    })

    test = pd.DataFrame({
        'x1': [200,300,200,500],
        'x2': [4,7,5,10],
        'x3': [16,11,15,20],
        'y':  [1,2,2,4]
    })

    x_train = train[['x1','x2','x3']]
    y_train = train['y']
    x_test = test[['x1','x2','x3']]
    y_test = test['y']

    clf = RidgeClassifier()
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    assert((y_pred==np.array([2,3,1,4])).all() == True)
    assert((np.array(y_test)==np.array([1,2,2,4])).all() == True)

    metrics_dict = classifier_scores(x_train=np.array(x_train), y_train=np.array(y_train), \
                  x_test=np.array(x_test), y_test=np.array(y_test), clf = clf)

    assert(np.isclose(metrics_dict['roc_auc'],0.541,rtol=0.05) == True)


# y_true has class 4 but y_pred don't
# and y_pred has class 3 but y_test does
def test_classifier_scores_multiclass_classnotpresent_ytrue_y_pred():
    train = pd.DataFrame({
        'x1': ['100','100','100','200','200','300'],
        'x2': [1,2,3,1,4,5],
        'x3': [4,8,12,3,12,9],
        'y':  [1,1,1,2,2,3]
    })

    test = pd.DataFrame({
        'x1': ['100','200','200'],
        'x2': [4,2,5],
        'x3': [16,6,13],
        'y':  [1,2,4]
    })

    x_train = train[['x1','x2','x3']]
    y_train = train['y']
    x_test = test[['x1','x2','x3']]
    y_test = test['y']

    clf = RidgeClassifier()
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    assert((y_pred==np.array([1,2,3])).all() == True)
    assert((np.array(y_test)==np.array([1,2,4])).all() == True)

    metrics_dict = classifier_scores(x_train=np.array(x_train), y_train=np.array(y_train), \
                  x_test=np.array(x_test), y_test=np.array(y_test), clf = clf)

    assert(np.isclose(metrics_dict['roc_auc'],1.0,rtol=0.05) == True)
