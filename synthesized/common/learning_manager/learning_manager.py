from typing import Optional, Dict, List, Union
import logging

import tensorflow as tf
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

from ..util import categorical_emd
from ...synthesizer import Synthesizer

logger = logging.getLogger(__name__)


class LearningManager:
    """This class will control the learning, checking that it improves and that stopping learning if necessary before
    the maximum number of iterations is reached.

    Example:
        Initialize the LearningManager, and in each iteration 'stop_learning' function should break the loop if True is
        returned:

        >>> lm = LearningManager()
        >>> for iteration in range(iterations):
        >>>     batch = get_batch()
        >>>     synthesizer.learn(batch)
        >>>     if lm.stop_learning(iteration, synthesizer=synthesizer, df_train=df_train):
        >>>         break

    """
    def __init__(self, check_frequency: int = 100, use_checkpointing: bool = True,
                 checkpoint_path: str = '/tmp/tf_checkpoints',
                 n_checks_no_improvement: int = 10, max_to_keep: int = 3,
                 patience: int = 750, must_reach_metric: float = None, good_enough_metric: float = None,
                 stop_metric_name: Optional[Union[str, List[str]]] = None):
        """Initialize LearningManager.

        Args:
            check_frequency: Frequency to perform checks.
            use_checkpointing: Whether to store checkpoints each 'check_frequency'.
            checkpoint_path: Directory where checkpoints will be saved.
            n_checks_no_improvement: If the LearningManager checks performance for 'no_learn_iterations' times without
                improvement, will return True and stop learning.
            max_to_keep: Defines the number of previous checkpoints stored.
            patience: How many iterations before start checking the performance.
            must_reach_metric: If this 'stop_metric' threshold is not reached, will always return False.
            good_enough_metric: If this 'stop_metric' threshold is not reached, will return True even if the model is
                still improving.
            stop_metric_name: Which 'stop_metric' will be evaluated in each iteration. If None, all 'stop_metric'es will
                be used, otherwise a str for a single 'stop_metric' or a list of str for more than one.
        """

        self.check_frequency = check_frequency
        self.use_checkpointing = use_checkpointing
        self.checkpoint_path = checkpoint_path
        self.n_checks_no_improvement = n_checks_no_improvement
        self.max_to_keep = max_to_keep
        self.patience = patience
        self.must_reach_metric = must_reach_metric
        self.good_enough_metric = good_enough_metric

        if stop_metric_name:
            if isinstance(stop_metric_name, str):
                assert stop_metric_name in ['ks_dist', 'corr', 'emd'], \
                    "Only 'ks_dist', 'corr', 'emd' supported, given stop_metric='{}'".format(stop_metric_name)
            elif isinstance(stop_metric_name, list):
                assert np.all([l in ['ks_dist', 'corr', 'emd'] for l in stop_metric_name]), \
                    "Only 'ks_dist', 'corr', 'emd' supported, given stop_metric='{}'".format(stop_metric_name)
        self.stop_metric_name = stop_metric_name

        self.count_no_improvement: int = 0
        self.best_stop_metric: Optional[float] = None
        self.best_checkpoint: Optional[str] = None
        self.best_iteration: int = 0

        if self.use_checkpointing:
            self.checkpoint = tf.train.Checkpoint()
            self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, directory=self.checkpoint_path,
                                                                 max_to_keep=self.max_to_keep)

    def stop_learning_check_metric(self, iteration: int, stop_metric: dict) -> bool:
        """Compare the 'stop_metric' against previous iteration, evaluate the criteria and return accordingly.

        Args:
            iteration: Iteration number.
            stop_metric: OrderedDict containing all stop_metrics.

        Returns
            bool: True if criteria are met to stop learning.
        """
        if iteration % self.check_frequency != 0:
            return False

        if self.stop_metric_name:
            if isinstance(self.stop_metric_name, str):
                stop_metric = {self.stop_metric_name: stop_metric[self.stop_metric_name]}
            if isinstance(self.stop_metric_name, list):
                stop_metric = {k: v for k, v in stop_metric.items() if k in self.stop_metric_name}

        total_stop_metric = np.nanmean(np.concatenate(list(stop_metric.values())))

        if np.isnan(total_stop_metric):
            logger.error("LearningManager :: Total 'stop_metric' is NaN")
            return False

        logger.debug("LearningManager :: Iteration {}. Current Metric = {:.4f}".format(iteration, total_stop_metric))

        if iteration < self.patience:
            return False
        if self.must_reach_metric and total_stop_metric > self.must_reach_metric:
            return False
        if self.good_enough_metric and total_stop_metric < self.good_enough_metric:
            return True

        if not self.best_stop_metric or total_stop_metric < self.best_stop_metric:
            self.best_stop_metric = total_stop_metric
            self.best_iteration = iteration
            self.count_no_improvement = 0
            if self.use_checkpointing:
                current_checkpoint = self.checkpoint_manager.save()
                logger.info("LearningManager :: New stop_metric minimum {:.4f} found at iteration {}, saving in '{}'"
                            .format(total_stop_metric, iteration, current_checkpoint))
                self.best_checkpoint = current_checkpoint
        else:
            self.count_no_improvement += 1
            if self.count_no_improvement >= self.n_checks_no_improvement:
                if self.use_checkpointing:
                    logger.info("LearningManager :: The model hasn't improved between iterations {1} and {0}. Restoring"
                                " model from iteration {1} with 'stop_metric' {2:.4f}".format(iteration,
                                                                                              self.best_iteration,
                                                                                              self.best_stop_metric))
                    # Restore best model and stop learning.
                    self.checkpoint.restore(self.best_checkpoint)
                else:
                    logger.info("LearningManager :: The model hasn't improved between iterations {1} and {0}. Stopping "
                                "learning procedure")
                return True

        return False

    def stop_learning_check_data(self, iteration: int, df_orig: pd.DataFrame, df_synth: pd.DataFrame,
                                 column_names: Optional[List[str]] = None) -> bool:
        """Given original an synthetic data, calculate the 'stop_metric' and compare it to previous iteration, evaluate
        the criteria and return accordingly.

        Args:
            iteration: Iteration number.
            df_orig: Original DataFrame.
            df_synth: Synthesized DataFrame.
            column_names: List of columns used to compute the 'break_metric'.

        Returns
            bool: True if criteria are met to stop learning.
        """
        if iteration % self.check_frequency != 0:
            return False

        stop_metrics = self._calculate_stop_metric_from_data(df_orig=df_orig, df_synth=df_synth,
                                                             column_names=column_names)
        return self.stop_learning_check_metric(iteration=iteration, stop_metric=stop_metrics)

    def stop_learning(self, iteration: int, synthesizer: Synthesizer, df_train: pd.DataFrame,
                      sample_size: int = None) -> bool:
        """Given a Synthesizer and the original data, get synthetic data, calculate the 'stop_metric', compare it to
        previous iteration, evaluate the criteria and return accordingly.

        Args:
            iteration: Iteration number.
            synthesizer: Synthesizer object, with 'synthesize(num_rows)' method.
            df_train: Original DataFrame.
            sample_size: Maximum sample size to compare DataFrames.

        Returns
            bool: True if criteria are met to stop learning.
        """
        if iteration % self.check_frequency != 0:
            return False

        sample_size = min(sample_size, len(df_train)) if sample_size else len(df_train)
        column_names = [v.name for v in synthesizer.get_values() if v.name in df_train.columns]
        df_synth = synthesizer.synthesize(num_rows=sample_size)
        return self.stop_learning_check_data(iteration, df_train.sample(sample_size), df_synth,
                                             column_names=column_names)

    def _calculate_stop_metric_from_data(self, df_orig: pd.DataFrame, df_synth: pd.DataFrame,
                                         column_names: Optional[List[str]] = None) -> Dict[str, List[float]]:
        """Calculate 'stop_metric' dictionary given two datasets. Each item in the dictionary will include a key
        (from self.stop_metric_name, allowed options are 'ks_dist', 'corr' and 'emd'), and a value (list of
        stop_metrics per column).

        Args
            df_orig: Original DataFrame.
            df_synth: Synthesized DataFrame.
            column_names: List of columns used to compute the 'break_metric'.

        Returns
            bool: True if criteria are met to stop learning.
        """
        if column_names is None:
            column_names_df: List[str] = df_orig.columns
        else:
            column_names_df = list(filter(lambda c: c in df_orig.columns, column_names))

        ks_distances = []
        emd = []

        for col in column_names_df:
            if df_orig[col].dtype.kind in ('f', 'i') and df_synth[col].dtype.kind in ('f', 'i'):
                ks_distances.append(ks_2samp(df_orig[col], df_synth[col])[0])
            else:
                try:
                    ks_distances.append(ks_2samp(
                        pd.to_numeric(pd.to_datetime(df_orig[col])),
                        pd.to_numeric(pd.to_datetime(df_synth[col]))
                    )[0])
                except ValueError:
                    emd.append(categorical_emd(df_orig[col].dropna(), df_synth[col].dropna()))

        corr = (df_orig[column_names].corr(method='kendall') -
                df_synth[column_names].corr(method='kendall')).abs().mean()

        stop_metrics = dict(
            ks_dist=list(ks_distances),
            corr=list(corr),
            emd=list(emd)
        )

        return stop_metrics
