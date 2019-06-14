import os

import pandas as pd
import pytest

from synthesized.core import BasicSynthesizer


@pytest.mark.integration
@pytest.mark.skip
def test_datasets_quick():
    passed = True
    failed = list()

    for root, dirs, files in os.walk('data'):
        for filename in files:
            if filename.startswith('_') or not filename.endswith('.csv'):
                continue

            try:
                data = pd.read_csv(os.path.join(root, filename))
                with BasicSynthesizer(data=data, capacity=8, depth=1, batch_size=8) as synthesizer:
                    synthesizer.learn(num_iterations=10, data=data)
                    synthesized = synthesizer.synthesize(n=10000)
                    assert len(synthesized) == 10000

            except Exception as exc:
                passed = False
                failed.append((os.path.join(root, filename), exc))

    assert passed, '\n\n' + '\n\n'.join('{}\n{}'.format(path, exc) for path, exc in failed) + '\n'
