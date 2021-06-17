import numpy as np
import pandas as pd
import pytest

from synthesized import HighDimSynthesizer
from synthesized.common.optimizers import DPOptimizer, Optimizer
from synthesized.config import HighDimConfig
from synthesized.metadata.factory import MetaExtractor


@pytest.mark.slow
def test_differential_privacy():
    n = 10000
    df_original = pd.DataFrame({'x': np.random.normal(size=n), 'y': np.random.choice(['a', 'b', 'c'], size=n)})

    df_meta = MetaExtractor.extract(df=df_original)
    config = HighDimConfig()
    config.differential_privacy = True
    config.delta = 1 / n
    config.epsilon = 1.0
    config.batch_size = 64
    config.learning_manager = False
    config.noise_multiplier = 1.0

    synthesizer = HighDimSynthesizer(df_meta=df_meta, config=config)
    assert isinstance(synthesizer._engine.optimizer, DPOptimizer)
    assert synthesizer._differential_privacy
    assert synthesizer._privacy_config.delta == 1 / n
    assert synthesizer._privacy_config.epsilon == 1.0
    assert synthesizer._privacy_config.noise_multiplier == 1.0

    synthesizer.learn(num_iterations=10, df_train=df_original)

    # can only do one training step if epsilon is zero.
    config.epsilon = 0.0
    synthesizer = HighDimSynthesizer(df_meta=df_meta, config=config)
    synthesizer.learn(num_iterations=10, df_train=df_original)
    assert synthesizer._global_step.numpy() == 1

    config.differential_privacy = False
    synthesizer = HighDimSynthesizer(df_meta=df_meta, config=config)
    assert synthesizer._differential_privacy is False
    assert type(synthesizer._engine.optimizer) == Optimizer
