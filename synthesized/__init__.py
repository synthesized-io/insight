import os
import sys
import warnings

if os.environ.get('SYNTHESIZED_TP_WARNINGS', 'false').lower() != 'true':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    for module in ['numpy', 'pandas', 'sklearn', 'tensorflow']:
        warnings.filterwarnings('ignore', module=module, append=True)

import synthesized.licence as _licence

from .common.synthesizer import Synthesizer
from .complex.conditional import ConditionalSampler
from .complex.data_imputer import DataImputer
from .complex.highdim import HighDimSynthesizer
from .complex.multi_table import TwoTableSynthesizer
from .metadata.data_frame_meta import DataFrameMeta
from .metadata.factory import MetaExtractor
from .version import __version__

try:
    _licence.verify()
except _licence.LicenceError as e:
    sys.exit(e)

__all__ = [
    '__version__', 'HighDimSynthesizer', 'ConditionalSampler', 'DataImputer', 'Synthesizer', 'DataFrameMeta',
    'MetaExtractor', 'TwoTableSynthesizer'
]
