import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier

from src.synthesized_insight.metrics import (
    Accuracy,
    ConfusionMatrix,
    F1Score,
    MeanAbsoluteError,
    MeanSquaredError,
    PRCurve,
    Precision,
    PredictiveModellingScore,
    R2Score,
    Recall,
    ROCCurve,
)
from src.synthesized_insight.metrics.modelling_metrics import (
    classifier_scores,
    classifier_scores_from_df,
    predictive_modelling_score,
    regressor_scores_from_df,
)

np.random.seed(42)


def test_predictive_modelling():
    x = np.random.normal(0, 1, 100)
    y = 2 * x + np.random.normal(0, 0.1, 100)
    y2 = 2.2 * x + np.random.normal(0, 0.1, 100)
    df_old = pd.DataFrame({'x': x, 'y': y})
    df_new = pd.DataFrame({'x': x, 'y': y2})

    pms = PredictiveModellingScore(model='Linear', y_label='y', x_labels=['x'], df_test=df_new)
    score, synth_score, _, _ = pms(df_old)
    assert score is not None
    assert synth_score is not None


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
    score, metric, task = predictive_modelling_score(data,
                                                     model='Logistic',
                                                     y_label=target,
                                                     x_labels=x_labels)
    assert metric == 'roc_auc'
    assert task == 'binary'


def test_predictive_modelling_score_rgr():
    n = 1000
    data = pd.DataFrame({
        'x1': np.random.randn(n),
        'x2': np.where(np.random.uniform(size=n) < 0.8,
                       np.random.randn(n), np.nan),
        'x3': np.random.choice(['a', 'b', 'c', 'd'], size=n),
        'y': np.random.randn(n)
    })

    target = 'y'
    score, metric, task = predictive_modelling_score(data,
                                                     model='Linear',
                                                     y_label=target,
                                                     x_labels=None)
    assert metric == 'r2_score'
    assert task == 'regression'


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
        'y': [1, 2, 3]
    })

    x_train = train[['x1', 'x2', 'x3']]
    y_train = train['y']
    x_test = test[['x1', 'x2', 'x3']]
    y_test = test['y']

    clf = RidgeClassifier()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    assert (y_pred == np.array([1, 2, 2])).all()
    assert (np.array(y_test) == np.array([1, 2, 3])).all()

    metrics_dict = classifier_scores(x_train=np.array(x_train),
                                     y_train=np.array(y_train),
                                     x_test=np.array(x_test),
                                     y_test=np.array(y_test), clf=clf)

    assert np.isclose(metrics_dict['roc_auc'], 0.75, rtol=0.05)


# y_true and y_pred both don't have class 4
def test_classifier_scores_multiclass_classnotpresent_both():
    train = pd.DataFrame({
        'x1': ['100', '100', '100', '200', '200', '300', '500'],
        'x2': [1, 2, 3, 1, 4, 5, 11],
        'x3': [4, 8, 12, 3, 12, 9, 4],
        'y': [1, 1, 1, 2, 2, 3, 4]
    })

    test = pd.DataFrame({
        'x1': ['100', '200', '200'],
        'x2': [4, 2, 5],
        'x3': [16, 6, 15],
        'y': [2, 3, 1]
    })

    x_train = train[['x1', 'x2', 'x3']]
    y_train = train['y']
    x_test = test[['x1', 'x2', 'x3']]
    y_test = test['y']

    clf = RidgeClassifier()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    assert (y_pred == np.array([1, 2, 3])).all()
    assert (np.array(y_test) == np.array([2, 3, 1])).all()

    metrics_dict = classifier_scores(x_train=np.array(x_train),
                                     y_train=np.array(y_train),
                                     x_test=np.array(x_test),
                                     y_test=np.array(y_test), clf=clf)

    assert np.isclose(metrics_dict['roc_auc'], 0.25, rtol=0.05)


# y_true doesn't have the class 3 but y_pred does
def test_classifier_scores_multiclass_classnotpresent_ytrue():
    train = pd.DataFrame({
        'x1': [100, 100, 100, 200, 200, 300, 500],
        'x2': [1, 2, 3, 1, 4, 5, 11],
        'x3': [4, 8, 12, 3, 12, 9, 22],
        'y': [1, 1, 1, 2, 2, 3, 4]
    })

    test = pd.DataFrame({
        'x1': [200, 300, 200, 500],
        'x2': [4, 7, 5, 10],
        'x3': [16, 11, 15, 20],
        'y': [1, 2, 2, 4]
    })

    x_train = train[['x1', 'x2', 'x3']]
    y_train = train['y']
    x_test = test[['x1', 'x2', 'x3']]
    y_test = test['y']

    clf = RidgeClassifier()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    assert (y_pred == np.array([2, 3, 1, 4])).all()
    assert (np.array(y_test) == np.array([1, 2, 2, 4])).all()

    metrics_dict = classifier_scores(x_train=np.array(x_train),
                                     y_train=np.array(y_train),
                                     x_test=np.array(x_test),
                                     y_test=np.array(y_test), clf=clf)

    assert np.isclose(metrics_dict['roc_auc'], 0.541, rtol=0.05)


# y_true has class 4 but y_pred don't
# and y_pred has class 3 but y_test does
def test_classifier_scores_multiclass_classnotpresent_ytrue_y_pred():
    train = pd.DataFrame({
        'x1': ['100', '100', '100', '200', '200', '300'],
        'x2': [1, 2, 3, 1, 4, 5],
        'x3': [4, 8, 12, 3, 12, 9],
        'y': [1, 1, 1, 2, 2, 3]
    })

    test = pd.DataFrame({
        'x1': ['100', '200', '200'],
        'x2': [4, 2, 5],
        'x3': [16, 6, 13],
        'y': [1, 2, 4]
    })

    x_train = train[['x1', 'x2', 'x3']]
    y_train = train['y']
    x_test = test[['x1', 'x2', 'x3']]
    y_test = test['y']

    clf = RidgeClassifier()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    assert (y_pred == np.array([1, 2, 3])).all()
    assert (np.array(y_test) == np.array([1, 2, 4])).all()

    metrics_dict = classifier_scores(x_train=np.array(x_train),
                                     y_train=np.array(y_train),
                                     x_test=np.array(x_test),
                                     y_test=np.array(y_test), clf=clf)

    assert np.isclose(metrics_dict['roc_auc'], 1.0, rtol=0.05)


def test_accuracy():
    y_true = [0, 1, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1, 1]
    acc = Accuracy()
    assert acc(y_true, y_true) == 1
    assert acc(y_true, y_pred) == 0.5


def test_prediction_scores_from_df():
    train = pd.DataFrame({
        'c1': [100, 100, 100, 200, 200, 300, 500],
        'c2': [1, 2, 3, 1, 4, 5, 11],
        'c3': [4, 8, 12, 3, 12, 9, 22],
        'c4': [0, 1, 1, 0, 1, 0, 0]
    })

    test = pd.DataFrame({
        'c1': [200, 300, 200, 500],
        'c2': [4, 7, 5, 10],
        'c3': [16, 11, 15, 20],
        'c4': [1, 0, 1, 0]
    })

    metrics_dict = classifier_scores_from_df(df_train=train,
                                             df_test=test,
                                             target='c4',
                                             clf=RidgeClassifier())

    assert 'roc_auc' in metrics_dict

    metrics_dict = regressor_scores_from_df(df_train=train,
                                            df_test=test,
                                            target='c2',
                                            rgr=Ridge())

    assert 'r2_score' in metrics_dict


def test_modelling_metrics():
    # regression
    target = [1, 2, 3, 4, 4, 6, 9, 1]
    r2_score = R2Score()
    assert r2_score(target, target) == 1

    mse = MeanSquaredError()
    assert mse(target, target) == 0

    mae = MeanAbsoluteError()
    assert mae(target, target) == 0

    # binary classification
    target = [1, 0, 1, 1, 1, 0, 0, 1]
    precision = Precision()
    assert precision(target, target) == 1

    recall = Recall()
    assert recall(target, target) == 1

    f1score = F1Score()
    assert f1score(target, target) == 1

    train = pd.DataFrame({
        'x1': ['100', '100', '100', '200', '200', '300', '500'],
        'x2': [1, 2, 3, 1, 4, 5, 11],
        'x3': [4, 8, 12, 3, 12, 9, 4],
        'y': [0, 1, 1, 1, 0, 0, 0]
    })

    test = pd.DataFrame({
        'x1': ['100', '200', '200'],
        'x2': [4, 2, 5],
        'x3': [16, 6, 15],
        'y': [1, 0, 1]
    })

    x_train = train[['x1', 'x2', 'x3']]
    y_train = train['y']
    x_test = test[['x1', 'x2', 'x3']]
    y_test = test['y']

    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    y_pred_proba = clf.predict_proba(x_test)

    roc_curve = ROCCurve()
    assert roc_curve(y_test, y_pred_proba=y_pred_proba[:, 1].reshape(-1, 1)) is not None

    pr_curve = PRCurve()
    assert pr_curve(y_test, y_pred_proba=y_pred_proba[:, 1].reshape(-1, 1)) is not None

    cf_matrix = ConfusionMatrix(True)
    assert cf_matrix(y_test, y_pred, y_pred_proba) is not None

    # multiclass classification
    target = ['a', 'c', 'a', 'b', 'b', 'b', 'c', 'a']
    precision = Precision(multiclass=True)
    assert precision(target, target) == 1

    recall = Recall(multiclass=True)
    assert recall(target, target) == 1

    f1score = F1Score(multiclass=True)
    assert f1score(target, target) == 1

    train = pd.DataFrame({
        'x1': ['100', '100', '100', '200', '200', '300', '500'],
        'x2': [1, 2, 3, 1, 4, 5, 11],
        'x3': [4, 8, 12, 3, 12, 9, 4],
        'y': [1, 1, 1, 2, 2, 3, 1]
    })

    test = pd.DataFrame({
        'x1': ['100', '200', '200'],
        'x2': [4, 2, 5],
        'x3': [16, 6, 15],
        'y': [2, 3, 1]
    })

    x_train = train[['x1', 'x2', 'x3']]
    y_train = train['y']
    x_test = test[['x1', 'x2', 'x3']]
    y_test = test['y']

    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    y_pred_proba = clf.predict_proba(x_test)

    roc_curve = ROCCurve(True)
    assert roc_curve(y_test, y_pred, y_pred_proba) is None

    pr_curve = PRCurve(True)
    assert pr_curve(y_test, y_pred, y_pred_proba) is None

    cf_matrix = ConfusionMatrix(True)
    assert cf_matrix(y_test, y_pred, y_pred_proba) is not None
