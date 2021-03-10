import logging
import os

import pandas as pd
import pytest
from scipy.stats import ks_2samp

from synthesized import HighDimSynthesizer
from synthesized.config import HighDimConfig
from synthesized.metadata_new.meta_builder import MetaExtractor
from tests.utils import progress_bar_testing

logger = logging.getLogger(__name__)


@pytest.mark.slow
@pytest.mark.skip
def test_datasets_quick():
    passed = True
    failed = list()

    for root, dirs, files in os.walk('data'):
        for filename in files:
            if filename.startswith('_') or not filename.endswith('.csv'):
                continue

            try:
                df_original = pd.read_csv(os.path.join(root, filename))
                config = HighDimConfig(capacity=8, num_layers=1, batch_size=8, learning_manager=False)
                df_meta = MetaExtractor.extract(df=df_original)
                with HighDimSynthesizer(df_meta=df_meta, config=config) as synthesizer:
                    synthesizer.learn(num_iterations=10, df_train=df_original)
                    df_synthesized = synthesizer.synthesize(num_rows=10000, progress_callback=progress_bar_testing)
                    assert len(df_synthesized) == 10000

            except Exception as exc:
                passed = False
                failed.append((os.path.join(root, filename), exc))

    assert passed, '\n\n' + '\n\n'.join('{}\n{}'.format(path, exc) for path, exc in failed) + '\n'


@pytest.mark.slow
def test_unittest_dataset_quick():
    df_original = pd.read_csv('data/unittest.csv')

    config = HighDimConfig(capacity=8, num_layers=1, batch_size=8, learning_manager=False)
    df_meta = MetaExtractor.extract(df=df_original)

    with HighDimSynthesizer(
        df_meta=df_meta, summarizer_dir='logs/', config=config
    ) as synthesizer:
        synthesizer.learn(num_iterations=10, df_train=df_original)
        df_synthesized = synthesizer.synthesize(num_rows=10000, progress_callback=progress_bar_testing)
        assert len(df_synthesized) == 10000


@pytest.mark.slow
def test_unittest_dataset():
    df_original = pd.read_csv('data/unittest.csv').dropna()
    df_meta = MetaExtractor.extract(df=df_original)
    conf = HighDimConfig(learning_manager=True)
    with HighDimSynthesizer(df_meta=df_meta, config=conf) as synthesizer:
        logger.info("LEARN")
        synthesizer.learn(num_iterations=None, df_train=df_original, callback_freq=0, callback=lambda a, b, c: False)
        logger.info("SYNTHESIZE")
        df_synthesized = synthesizer.synthesize(num_rows=len(df_original), progress_callback=progress_bar_testing)
        assert len(df_synthesized) == len(df_original)

    distances = [
        (name, ks_2samp(df_original[name], df_synthesized[name])[0])
        for name in df_original.keys()
    ]
    assert all(distance < 0.33 for _, distance in distances), 'Failed: ' + ', '.join(
        '{}={:.3f}'.format(name, distance) for name, distance in distances if distance >= 0.33
    )
