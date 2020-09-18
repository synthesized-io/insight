import pandas as pd
import pytest

from synthesized.insight.fairness import FairnessScorer


def test_fairness_scorer():
    data = pd.read_csv('data/templates/claim_prediction.csv')
    sensitive_attributes = ["age", "sex", "children", "region"]
    target = "insuranceclaim"

    fairness_scorer = FairnessScorer(data, sensitive_attrs=sensitive_attributes, target=target)
    dist_score, dist_biases = fairness_scorer.distributions_score()
    clf_score, clf_biases = fairness_scorer.classification_score()

    assert 0. <= dist_score <= 1.
    assert 0. <= clf_score <= 1.


@pytest.mark.slow
def test_fairness_scorer_detect_sensitive():
    data = pd.read_csv('data/templates/claim_prediction.csv')
    target = "insuranceclaim"

    fairness_scorer = FairnessScorer.init_detect_sensitive(data, target=target)
    assert fairness_scorer.get_sensitive_attrs() == ["age", "sex", "children", "region"]


@pytest.mark.slow
def test_fairness_scorer_no_sensitive_attrs():
    data = pd.read_csv('data/templates/claim_prediction.csv')
    sensitive_attributes = []
    target = "insuranceclaim"

    fairness_scorer = FairnessScorer(data, sensitive_attrs=sensitive_attributes, target=target)
    dist_score, dist_biases = fairness_scorer.distributions_score()
    clf_score, clf_biases = fairness_scorer.classification_score()

    assert dist_score == 0. and len(dist_biases) == 0
    assert clf_score == 0. and len(clf_biases) == 0
