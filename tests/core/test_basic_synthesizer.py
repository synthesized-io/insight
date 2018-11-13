import numpy as np
import pytest
from scipy.stats import ks_2samp

from loaders import credit
from synthesized.core import BasicSynthesizer


@pytest.mark.integration
def test_james_dataset_generation():
    data = credit.load_data()
    with BasicSynthesizer(data=data, exclude_encoding_loss=True) as synthesizer:
        synthesizer.learn(data=data, num_iterations=20000, verbose=1000)
        synthesized = synthesizer.synthesize(n=len(data))
    synthesized = synthesizer.preprocess(synthesized)
    data = synthesizer.preprocess(data.copy())
    avg_distance = np.mean([ks_2samp(data[col], synthesized[col])[0] for col in data.columns])
    assert avg_distance < 0.2
