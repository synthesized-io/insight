from typing import Optional, Dict, List, Union
import logging

import tensorflow as tf
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, kendalltau

from ..util import categorical_emd
from ...synthesizer import Synthesizer

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s :: %(levelname)s :: %(message)s', level=logging.INFO)


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
                 n_checks_no_improvement: int = 10, max_to_keep: int = 3, patience: int = 750,
                 tol: float = 1e-4, must_reach_metric: float = None, good_enough_metric: float = None,
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
            tol: Tolerance for comparing current 'stop_metric' to best 'stop metric'.
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
        self.tol = tol
        self.must_reach_metric = must_reach_metric
        self.good_enough_metric = good_enough_metric

        allowed_stop_metric_names = ['ks_distances', 'corr_distances', 'emd_distances']
        if stop_metric_name:
            if isinstance(stop_metric_name, str):
                assert stop_metric_name in allowed_stop_metric_names, \
                    "Only {} supported, given stop_metric='{}'".format(stop_metric_name, allowed_stop_metric_names)
            elif isinstance(stop_metric_name, list):
                assert np.all([l in allowed_stop_metric_names for l in stop_metric_name]), \
                    "Only {} supported, given stop_metric='{}'".format(stop_metric_name, allowed_stop_metric_names)
        self.stop_metric_name = stop_metric_name

        self.count_no_improvement: int = 0
        self.best_stop_metric: Optional[float] = None
        self.best_checkpoint: Optional[str] = None
        self.best_iteration: int = 0

        if self.use_checkpointing:
            self.checkpoint = tf.train.Checkpoint()
            self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, directory=self.checkpoint_path,
                                                                 max_to_keep=self.max_to_keep)

        # Some constants to compute metrics
        self.max_sample_dates = 2500
        self.num_unique_categorical = 100
        self.max_pval = 0.05
        self.nan_fraction_threshold = 0.25
        self.non_nan_count_threshold = 500
        self.categorical_threshold_log_multiplier = 2.5

    def stop_learning_check_metric(self, iteration: int, stop_metric: Dict[str, List[float]]) -> bool:
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

        if not self.best_stop_metric or total_stop_metric < self.best_stop_metric - self.tol:
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
        column_names = [col for v in synthesizer.get_values() for col in v.learned_input_columns()]
        if len(column_names) == 0:
            return False

        if hasattr(synthesizer, 'conditions') and len(synthesizer.conditions) > 0:
            raise NotImplementedError

        df_synth = synthesizer.synthesize(num_rows=sample_size)
        return self.stop_learning_check_data(iteration, df_train.sample(sample_size), df_synth,
                                             column_names=column_names)

    def restart_learning_manager(self):
        self.count_no_improvement: int = 0
        self.best_stop_metric: Optional[float] = None
        self.best_checkpoint: Optional[str] = None
        self.best_iteration: int = 0

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

        # Calculate distances for all columns
        ks_distances = []
        emd_distances = []
        corr_distances = []
        numerical_columns = []

        len_test = len(df_orig)
        for col in column_names_df:

            if df_orig[col].dtype.kind == 'f':
                col_test_clean = df_orig[col].dropna()
                col_synth_clean = df_synth[col].dropna()
                if len(col_test_clean) < len_test:
                    logger.debug("Column '{}' contains NaNs. Computing KS distance with {}/{} samples"
                                 .format(col, len(col_test_clean), len_test))
                ks_distance, _ = ks_2samp(col_test_clean, col_synth_clean)
                ks_distances.append(ks_distance)
                numerical_columns.append(col)

            elif df_orig[col].dtype.kind == 'i':
                if df_orig[col].nunique() < np.log(len(df_orig)) * self.categorical_threshold_log_multiplier:
                    logger.debug(
                        "Column '{}' treated as categorical with {} categories".format(col, df_orig[col].nunique()))
                    emd_distance = categorical_emd(df_orig[col].dropna(), df_synth[col].dropna())
                    emd_distances.append(emd_distance)

                else:
                    col_test_clean = df_orig[col].dropna()
                    col_synth_clean = df_synth[col].dropna()
                    if len(col_test_clean) < len_test:
                        logger.debug("Column '{}' contains NaNs. Computing KS distance with {}/{} samples"
                                     .format(col, len(col_test_clean), len_test))
                    ks_distance, _ = ks_2samp(col_test_clean, col_synth_clean)
                    ks_distances.append(ks_distance)

                numerical_columns.append(col)

            elif df_orig[col].dtype.kind in ('O', 'b'):

                # Try to convert to numeric
                col_num = pd.to_numeric(df_orig[col], errors='coerce')
                if col_num.isna().sum() / len(col_num) < self.nan_fraction_threshold:
                    df_orig[col] = col_num
                    df_synth[col] = pd.to_numeric(df_synth[col], errors='coerce')
                    numerical_columns.append(col)

                # if (is not sampling) and (is not date):
                elif (df_orig[col].nunique() <= np.log(len(df_orig)) * self.categorical_threshold_log_multiplier) and \
                        np.all(pd.to_datetime(df_orig[col].sample(min(len(df_orig), self.max_sample_dates)),
                                              errors='coerce').isna()):
                    emd_distance = categorical_emd(df_orig[col].dropna(), df_synth[col].dropna())
                    if not np.isnan(emd_distance):
                        emd_distances.append(emd_distance)

        for i in range(len(numerical_columns)):
            col_i = numerical_columns[i]
            for j in range(i + 1, len(numerical_columns)):
                col_j = numerical_columns[j]
                test_clean = df_orig[[col_i, col_j]].dropna()
                synth_clean = df_synth[[col_i, col_j]].dropna()

                if len(test_clean) > 0 and len(synth_clean) > 0:
                    corr_orig, pvalue_orig = kendalltau(test_clean[col_i].values, test_clean[col_j].values)
                    corr_synth, pvalue_synth = kendalltau(synth_clean[col_i].values, synth_clean[col_j].values)

                    if pvalue_orig <= self.max_pval or pvalue_synth <= self.max_pval:
                        corr_distances.append(abs(corr_orig - corr_synth))

        stop_metrics = {
            'ks_distances': ks_distances,
            'corr_distances': corr_distances,
            'emd_distances': emd_distances
        }

        return stop_metrics
