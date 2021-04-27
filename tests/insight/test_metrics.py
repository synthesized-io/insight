import logging

import numpy as np
import pandas as pd

from sklearn.linear_model import RidgeClassifier
from synthesized.insight.metrics import categorical_logistic_correlation, kendell_tau_correlation
from synthesized.insight.metrics.modelling_metrics import classifier_scores

logger = logging.getLogger(__name__)


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
    assert value is not None # continuous, categorical -> not None


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