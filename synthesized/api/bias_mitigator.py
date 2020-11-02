from typing import List, Callable

import pandas as pd

from .fairness import FairnessScorer
from .synthesizer import Synthesizer

from ..insight.fairness import BiasMitigator as _BiasMitigator


class BiasMitigator:
    """Find distribution biases in data and generate the inverse data to mitigate these biases.
    """

    def __init__(self, synthesizer: Synthesizer, fairness_scorer: FairnessScorer):
        """Given a FairnessScorer, build a Bias Mitigator.

        Args:
            synthesizer: An underlying synthesizer.
            fairness_scorer: The fairness score to compute distribution biases from.
        """
        self._bias_mitigator = _BiasMitigator(synthesizer=synthesizer._synthesizer,
                                              fairness_scorer=fairness_scorer._fairness_scorer)

    @classmethod
    def from_dataframe(cls, synthesizer: Synthesizer, df: pd.DataFrame, target: str,
                       sensitive_attrs: List[str]) -> 'BiasMitigator':
        """Given a DataFrame, build a Bias Mitigator.

        Args:
            synthesizer: An underlying synthesizer.
            df: Pandas DataFrame containing the data to mitigate biases.
            target: Name of the column containing the target feature.
            sensitive_attrs: Given sensitive attributes.
        """
        fairness_scorer = FairnessScorer(df, sensitive_attrs=sensitive_attrs, target=target)
        bias_mitigator = cls(synthesizer=synthesizer, fairness_scorer=fairness_scorer)
        return bias_mitigator

    def mitigate_biases_by_chunks(self, df: pd.DataFrame, chunk_size: int = 5, marginal_softener: float = 0.2,
                                  n_loops: int = 20, progress_callback: Callable[[int], None] = None) -> pd.DataFrame:
        """Mitigate biases iteratively in chunks.

        Args:
            df: Pandas DataFrame containing the data to mitigate biases.
            chunk_size: Number of biases to mitigate each iteration.
            marginal_softener: Whether to mitigate each bias completely (1.) or just a proportion of it
            bias_min_dist: Minimum distance to be considered bias.
            n_loops: Maximum number of loops to try to mitigate biases.
            produce_nans:  Whether the output DF contains NaNs.
            progress_callback: Progress bar callback.
        """
        return self._bias_mitigator.mitigate_biases_by_chunks(
            df=df, chunk_size=chunk_size, marginal_softener=marginal_softener, n_loops=n_loops,
            progress_callback=progress_callback
        )
