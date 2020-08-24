from typing import Union, Callable, Optional, Tuple, Dict

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
                 *explicit_marginals: Tuple[str, Dict[str, float]],
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
        """Trains the underlying generative model for the given iterations.

        Repeated calls continue training the model, possibly on different data.

        Args:
            df_train: The training data.
            num_iterations: The number of training iterations (not epochs).
            callback: A callback function, e.g. for logging purposes. Takes the synthesizer
                instance, the iteration number, and a dictionary of values (usually the losses) as
                arguments. Aborts training if the return value is True.
            callback_freq: Callback frequency.

        """
        self._conditional_sampler.learn(
            df_train=df_train, num_iterations=num_iterations, callback=callback, callback_freq=callback_freq
        )

    def synthesize(self,
                   num_rows: int,
                   conditions: Union[dict, pd.DataFrame] = None,
                   progress_callback: Callable[[int], None] = None) -> pd.DataFrame:
        """Generate the given number of new data rows according to the ConditionalSynthesizer's explicit marginals.

        Args:
            num_rows: The number of rows to generate.
            conditions: The condition values for the generated rows.
            progress_callback: Progress bar callback.

        Returns:
            The generated data.

        """
        return self._conditional_sampler.synthesize(
            num_rows=num_rows, conditions=conditions, progress_callback=progress_callback
        )
