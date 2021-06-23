import pytest

from synthesized.common.optimizers import DPOptimizer, Optimizer


def test_set_optimizer():

    with pytest.raises(ValueError):
        opt = Optimizer('op', optimizer='sgd', learning_rate=1e-3)

    opt = Optimizer('op', optimizer='adam', learning_rate=1e-3)
    opt = Optimizer('op', optimizer='adadelta', learning_rate=1e-3)

    with pytest.raises(ValueError):
        opt = DPOptimizer('op', optimizer='adadelta', learning_rate=1e-3)

    opt = DPOptimizer('op', optimizer='adam', learning_rate=1e-3)
    opt = DPOptimizer('op', optimizer='adagrad', learning_rate=1e-3)
    opt = DPOptimizer('op', optimizer='sgd', learning_rate=1e-3)
