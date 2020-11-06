import pandas as pd
import pytest

from synthesized.insight.fairness import FairnessScorer
from synthesized.insight.fairness.fairness_scorer import VariableType
from synthesized.testing.utils import testing_progress_bar


@pytest.mark.slow
@pytest.mark.parametrize(
    "file_name,sensitive_attributes,target",
    [
        pytest.param("data/credit_with_categoricals.csv", ["age"], "SeriousDlqin2yrs", id="binary_target"),
        pytest.param("data/credit_with_categoricals.csv", ["age"], "RevolvingUtilizationOfUnsecuredLines",
                     id="continuous_target"),
        pytest.param("data/credit_with_categoricals.csv", ["age"], "effort", id="multiple_categories_target"),
        pytest.param("data/templates/claim_prediction.csv", ["age", "sex", "children", "region"], "insuranceclaim",
                     id="claim_prediction"),
        pytest.param("data/templates/claim_prediction.csv", [], "insuranceclaim", id="no_sensitive_attrs"),
    ]
)
def test_fairness_scorer_parametrize(file_name, sensitive_attributes, target):

    data = pd.read_csv(file_name)
    sample_size = 10_000
    data = data.sample(sample_size) if len(data) > sample_size else data

    fairness_scorer = FairnessScorer(data, sensitive_attrs=sensitive_attributes, target=target)

    # Distributions Score
    dist_score, dist_biases = fairness_scorer.distributions_score(progress_callback=testing_progress_bar)

    assert 0. <= dist_score <= 1.
    assert not any([dist_biases[c].isna().any() for c in dist_biases.columns])

    # Classification score
    if fairness_scorer.target_variable_type == VariableType.Binary:
        clf_score, clf_biases = fairness_scorer.classification_score(progress_callback=testing_progress_bar)

        assert 0. <= clf_score <= 1.
        assert not any([dist_biases[c].isna().any() for c in clf_biases.columns])


@pytest.mark.slow
def test_fairness_scorer_detect_sensitive():
    data = pd.read_csv("data/templates/claim_prediction.csv")
    target = "insuranceclaim"

    fairness_scorer = FairnessScorer.init_detect_sensitive(data, target=target)
    assert fairness_scorer.get_sensitive_attrs() == ["age", "sex", "children", "region"]

    dist_score, dist_biases = fairness_scorer.distributions_score(progress_callback=testing_progress_bar)
    clf_score, clf_biases = fairness_scorer.classification_score(progress_callback=testing_progress_bar)

    assert 0. <= dist_score <= 1.
    assert 0. <= clf_score <= 1.

    assert not any([dist_biases[c].isna().any() for c in dist_biases.columns])
    assert not any([dist_biases[c].isna().any() for c in clf_biases.columns])


@pytest.mark.fast
def test_detect_sensitive():
    attrs = ["i", "love", "sex", "in", "any", "location"]

    sensitive_attrs = FairnessScorer.detect_sensitive_attrs(attrs)
    assert sensitive_attrs == ["sex", "location"]
