from typing import Dict, Any, Union, List
import logging

import pandas as pd
import simplejson

from ...highdim import HighDimSynthesizer
from .sanitizer import Sanitizer

logger = logging.getLogger(__name__)


def privacy_check(data: pd.DataFrame, num_iterations: int = None, synthesizer_class: str = 'HighDimSynthesizer',
                  n_cols: int = 3, distance_step: float = 1e-3, synthesizer_params: Dict[str, Any] = None,
                  return_json: bool = True) -> Union[str, List[Dict]]:

    data = data.copy()
    data.dropna(inplace=True)

    if synthesizer_params is None:
        synthesizer_params = dict()

    if synthesizer_class == 'HighDimSynthesizer':
        synthesizer = HighDimSynthesizer(df=data, **synthesizer_params)
        synthesizer.__enter__()
        synthesizer.learn(df_train=data, num_iterations=num_iterations)
    else:
        raise NotImplementedError

    sanitizer = Sanitizer(synthesizer, data, distance_step)
    learned_columns = sanitizer.learned_columns

    import time

    results = []
    for i in range(n_cols):
        d: Dict[str, Any] = dict()
        n_cols = len(learned_columns) - i
        synthesized = synthesizer.synthesize(num_rows=len(data))
        t_start = time.perf_counter()
        df_synthesized = sanitizer._sanitize(synthesized, n_cols=n_cols)
        took = time.perf_counter() - t_start
        logger.debug('took {:.2f}s'.format(took))

        d['n_cols'] = n_cols
        d['n_categorical'] = len(sanitizer.categorical_values)
        d['results'] = {
            'cleaned-rows-rate': (len(data) - len(df_synthesized)) / len(data),
            'cleaned-rows-num': len(data) - len(df_synthesized),
            'total-rows': len(data),
        }
        results.append(d)

    if return_json:
        return simplejson.dumps(results, indent='\t')
    else:
        return results
