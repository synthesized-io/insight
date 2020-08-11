from typing import Union, Callable, Optional, Tuple, Dict, Any

import pandas as pd

from .synthesizer import Synthesizer
from ..complex.conditional import ConditionalSampler as _ConditionalSampler


class ConditionalSampler(Synthesizer):
    """Samples from the synthesizer conditionally on explicitly defined marginals of some columns.

        Example:
            >>> cond = ConditionalSampler(synthesizer, ('SeriousDlqin2yrs', {'0': 0.3, '1': 0.7}),
            >>>                                        ('age', {'[0.0, 50.0)': 0.5, '[50.0, 100.0)': 0.5}))
            >>> cond.synthesize(num_rows=10))
    """

    def __init__(self,
                 synthesizer: Synthesizer,
                 *explicit_marginals: Tuple[str, Dict[Any, float]],
                 min_sampled_ratio: float = 0.001,
                 synthesis_batch_size: Optional[int] = 16384):
        """Create ConditionalSampler.

        Args:
            synthesizer: An underlying synthesizer
            *explicit_marginals: A dict of desired marginal distributions per column.
                Distributions defined as density per category or bin. The result will be sampled
                from the synthesizer conditionally on these marginals.
            min_sampled_ratio: Stop synthesis if ratio of successfully sampled records is less than given value.
            synthesis_batch_size: Synthesis batch size
        """
        super().__init__()
        self._conditional_sampler = _ConditionalSampler(
            synthesizer._synthesizer, *explicit_marginals,
            min_sampled_ratio=min_sampled_ratio, synthesis_batch_size=synthesis_batch_size
        )

    def learn(self, df_train: pd.DataFrame, num_iterations: Optional[int],
              callback: Callable[[object, int, dict], bool] = None, callback_freq: int = 0) -> None:
        self._conditional_sampler.learn(
            df_train=df_train, num_iterations=num_iterations, callback=callback, callback_freq=callback_freq
        )

    def synthesize(self,
                   num_rows: int,
                   conditions: Union[dict, pd.DataFrame] = None,
                   progress_callback: Callable[[int], None] = None) -> pd.DataFrame:

        return self._conditional_sampler.synthesize(
            num_rows=num_rows, conditions=conditions, progress_callback=progress_callback
        )
