import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier

from src.synthesized_insight.metrics import PredictiveModellingScore
from src.synthesized_insight.metrics.modelling_metrics import (classifier_scores,
                                                               predictive_modelling_score)


np.random.seed(42)


def test_predictive_modelling():
    pms = PredictiveModellingScore(model='Linear', y_label='y', x_labels=['x'])

    x = np.random.normal(0, 1, 100)
    y = 2 * x + np.random.normal(0, 0.1, 100)
    y2 = 2.2 * x + np.random.normal(0, 0.1, 100)
    df_old = pd.DataFrame({'x': x, 'y': y})
    df_new = pd.DataFrame({'x': x, 'y': y2})

    assert pms(df_old) is not None
    assert pms(df_old, df_new) is not None


def test_predictive_modelling_score_clf():
    n = 25
    data = pd.DataFrame({
        'x1': np.random.randn(n),
        'x2': np.where(np.random.uniform(size=n) < 0.8,
                       np.random.randn(n), np.nan),
        'x3': np.random.choice(['a', 'b', 'c', 'd'], size=n),
        'x4': np.where(np.random.uniform(size=n) < 0.8,
                       np.random.randint(1e5, size=n).astype(str), ''),
        'y': np.random.choice([0, 1], size=n)
    })

    target = 'y'
    x_labels = list(filter(lambda c: c != target, data.columns))
    predictive_modelling_score(data,
                               model='Logistic',
                               y_label=target,
                               x_labels=x_labels)


def test_predictive_modelling_score_rgr():
    n = 1000
    data = pd.DataFrame({
        'x1': np.random.randn(n),
        'x2': np.where(np.random.uniform(size=n) < 0.8,
                       np.random.randn(n), np.nan),
        'x3': np.random.choice(['a', 'b', 'c', 'd'], size=n),
        'y': np.random.choice([0, 1], size=n)
    })

    target = 'y'
    x_labels = list(filter(lambda c: c != target, data.columns))
    predictive_modelling_score(data,
                               model='Linear',
                               y_label=target,
                               x_labels=x_labels)


# y_true has class 3, but y_pred doesn't
def test_classifier_scores_multiclass_classnotpresent_ypred():
    train = pd.DataFrame({
        'x1': ['100', '100', '100', '200', '200', '300'],
        'x2': [1, 2, 3, 1, 4, 5],
        'x3': [4, 8, 12, 3, 12, 9],
        'y': [1, 1, 1, 2, 2, 3]
    })

    test = pd.DataFrame({
        'x1': ['100', '200', '200'],
        'x2': [4, 2, 5],
        'x3': [16, 6, 15],
        'y':  [1, 2, 3]
    })

    x_train = train[['x1', 'x2', 'x3']]
    y_train = train['y']
    x_test = test[['x1', 'x2', 'x3']]
    y_test = test['y']

    clf = RidgeClassifier()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    assert (y_pred==np.array([1,2,2])).all() == True
    assert (np.array(y_test)==np.array([1,2,3])).all() == True

    metrics_dict = classifier_scores(x_train=np.array(x_train),
                                     y_train=np.array(y_train),
                                     x_test=np.array(x_test),
                                     y_test=np.array(y_test), clf=clf)

    assert np.isclose(metrics_dict['roc_auc'],0.75,rtol=0.05) == True


# y_true and y_pred both don't have class 4
def test_classifier_scores_multiclass_classnotpresent_both():
    train = pd.DataFrame({
        'x1': ['100', '100', '100', '200', '200', '300', '500'],
        'x2': [1, 2, 3, 1, 4, 5, 11],
        'x3': [4, 8, 12, 3, 12, 9, 4],
        'y':  [1, 1, 1, 2, 2, 3, 4]
    })

    test = pd.DataFrame({
        'x1': ['100', '200', '200'],
        'x2': [4, 2, 5],
        'x3': [16, 6, 15],
        'y':  [2, 3, 1]
    })

    x_train = train[['x1', 'x2', 'x3']]
    y_train = train['y']
    x_test = test[['x1', 'x2', 'x3']]
    y_test = test['y']

    clf = RidgeClassifier()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    assert (y_pred==np.array([1,2,3])).all() == True
    assert (np.array(y_test)==np.array([2,3,1])).all() == True

    metrics_dict = classifier_scores(x_train=np.array(x_train),
                                     y_train=np.array(y_train),
                                     x_test=np.array(x_test),
                                     y_test=np.array(y_test), clf= clf)

    assert np.isclose(metrics_dict['roc_auc'],0.25,rtol=0.05) == True


# y_true doesn't have the class 3 but y_pred does
def test_classifier_scores_multiclass_classnotpresent_ytrue():
    train = pd.DataFrame({
        'x1': [100, 100, 100, 200, 200, 300, 500],
        'x2': [1, 2, 3, 1, 4, 5, 11],
        'x3': [4, 8, 12, 3, 12, 9, 22],
        'y':  [1, 1, 1, 2, 2, 3, 4]
    })

    test = pd.DataFrame({
        'x1': [200, 300, 200, 500],
        'x2': [4, 7, 5, 10],
        'x3': [16, 11, 15, 20],
        'y':  [1, 2, 2, 4]
    })

    x_train = train[['x1', 'x2', 'x3']]
    y_train = train['y']
    x_test = test[['x1', 'x2', 'x3']]
    y_test = test['y']

    clf = RidgeClassifier()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    assert (y_pred==np.array([2,3,1,4])).all() == True
    assert (np.array(y_test)==np.array([1,2,2,4])).all() == True

    metrics_dict = classifier_scores(x_train=np.array(x_train),
                                     y_train=np.array(y_train),
                                     x_test=np.array(x_test),
                                     y_test=np.array(y_test), clf=clf)

    assert np.isclose(metrics_dict['roc_auc'], 0.541, rtol=0.05) == True


# y_true has class 4 but y_pred don't
# and y_pred has class 3 but y_test does
def test_classifier_scores_multiclass_classnotpresent_ytrue_y_pred():
    train = pd.DataFrame({
        'x1': ['100', '100', '100', '200', '200', '300'],
        'x2': [1, 2, 3, 1, 4, 5],
        'x3': [4, 8, 12, 3, 12, 9],
        'y':  [1, 1, 1, 2, 2, 3]
    })

    test = pd.DataFrame({
        'x1': ['100', '200', '200'],
        'x2': [4, 2, 5],
        'x3': [16, 6, 13],
        'y':  [1, 2, 4]
    })

    x_train = train[['x1', 'x2', 'x3']]
    y_train = train['y']
    x_test = test[['x1', 'x2', 'x3']]
    y_test = test['y']

    clf = RidgeClassifier()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    assert (y_pred==np.array([1,2,3])).all() == True
    assert (np.array(y_test)==np.array([1,2,4])).all() == True

    metrics_dict = classifier_scores(x_train=np.array(x_train),
                                     y_train=np.array(y_train),
                                     x_test=np.array(x_test),
                                     y_test=np.array(y_test), clf=clf)

    assert np.isclose(metrics_dict['roc_auc'], 1.0, rtol=0.05) == True