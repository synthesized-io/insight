import os
import pytest

import pandas as pd
import sklearn as skl
from scipy.stats import ks_2samp

from synthesized import HighDimSynthesizer


@pytest.mark.integration
@pytest.mark.skip
def test_datasets_quick():
    passed = True
    failed = list()
    print(skl.__version__)

    for root, dirs, files in os.walk('data'):
        for filename in files:
            if filename.startswith('_') or not filename.endswith('.csv'):
                continue

            try:
                df_original = pd.read_csv(os.path.join(root, filename))
                with HighDimSynthesizer(
                    df=df_original, capacity=8, num_layers=1, batch_size=8
                ) as synthesizer:
                    synthesizer.learn(num_iterations=10, df_train=df_original)
                    df_synthesized = synthesizer.synthesize(num_rows=10000)
                    assert len(df_synthesized) == 10000

            except Exception as exc:
                passed = False
                failed.append((os.path.join(root, filename), exc))

    assert passed, '\n\n' + '\n\n'.join('{}\n{}'.format(path, exc) for path, exc in failed) + '\n'


def test_unittest_dataset_quick():
    df_original = pd.read_csv('data/unittest.csv')
    with HighDimSynthesizer(
        df=df_original, capacity=8, num_layers=1, batch_size=8, condition_columns=['SeriousDlqin2yrs'],
        learning_manager=False, summarizer_dir='logs/'
    ) as synthesizer:
        synthesizer.learn(num_iterations=10, df_train=df_original)
        df_synthesized = synthesizer.synthesize(num_rows=10000, conditions={'SeriousDlqin2yrs': 1})
        assert len(df_synthesized) == 10000
        assert (df_synthesized['SeriousDlqin2yrs'] == 1).all()


@pytest.mark.integration
def test_unittest_dataset():
    df_original = pd.read_csv('data/unittest.csv').dropna()
    with HighDimSynthesizer(df=df_original) as synthesizer:
        synthesizer.learn(num_iterations=5000, df_train=df_original)
        df_synthesized = synthesizer.synthesize(num_rows=len(df_original))
        assert len(df_synthesized) == len(df_original)

    distances = [
        (name, ks_2samp(df_original[name], df_synthesized[name])[0])
        for name in df_original.keys()
    ]
    assert all(distance < 0.33 for _, distance in distances), 'Failed: ' + ', '.join(
        '{}={:.3f}'.format(name, distance) for name, distance in distances if distance >= 0.33
    )
