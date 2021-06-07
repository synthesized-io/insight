import pytest

from synthesized.complex.unifier import EnsembleUnifier
from synthesized.testing.unifier_evaluation import evaluate_unifier


@pytest.mark.slow
def test_evaluate_unifier():
    result = evaluate_unifier(EnsembleUnifier, "tests/testing/simple_test.yaml")
    assert "simple_test" in result
    assert "HighDim_0" not in result
