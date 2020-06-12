import os
import warnings

if os.environ.get('SYNTHESIZED_TP_WARNINGS', 'false').lower() != 'true':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    for module in ['numpy', 'pandas', 'sklearn', 'tensorflow']:
        warnings.filterwarnings('ignore', module=module, append=True)

from .common import Synthesizer  # noqa: F402
from .highdim import HighDimSynthesizer  # noqa: F402
from .metadata import DataPanel, MetaExtractor  # noqa: F402
from .scenario import ScenarioSynthesizer  # noqa: F402
from .series import SeriesSynthesizer  # noqa: F402
from .version import __version__  # noqa: F402

__all__ = [
    '__version__', 'HighDimSynthesizer', 'ScenarioSynthesizer', 'SeriesSynthesizer', 'Synthesizer', 'DataPanel'
]
