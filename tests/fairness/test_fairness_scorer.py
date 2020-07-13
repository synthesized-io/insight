import pandas as pd

from synthesized.fairness import FairnessScorer


def test_fairness_scorer():
    data = pd.read_csv('data/templates/claim_prediction.csv')
    sensitive_attributes = ["age", "sex", "children", "region"]
    target = "insuranceclaim"

    fairness_scorer = FairnessScorer(data, sensitive_attrs=sensitive_attributes, target=target)
    dist_score, dist_biases = fairness_scorer.distributions_score()
    clf_score, clf_biases = fairness_scorer.classification_score()

    assert 0. <= dist_score <= 1.
    assert 0. <= clf_score <= 1.
