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
                 patience: int = 750, must_reach_loss: float = None, good_enough_loss: float = None,
                 loss_name: Optional[Union[str, List[str]]] = None):
        """Initialize LearningManager.

        Args:
            check_frequency: Frequency to perform checks.
            use_checkpointing: Whether to store checkpoints each 'check_frequency'.
            checkpoint_path: Directory where checkpoints will be saved.
            n_checks_no_improvement: If the LearningManager checks performance for 'no_learn_iterations' times without
                improvement, will return True and stop learning.
            max_to_keep: Defines the number of previous checkpoints stored.
            patience: How many iterations before start checking the performance.
            must_reach_loss: If this loss threshold is not reached, will always return False.
            good_enough_loss: If this loss threshold is not reached, will return True even if the model is still
                improving.
            loss_name: Which loss will be evaluated in each iteration. If None, all losses will be used, otherwise
                a str for a single loss or a list of str for more than one.
        """

        self.check_frequency = check_frequency
        self.use_checkpointing = use_checkpointing
        self.checkpoint_path = checkpoint_path
        self.n_checks_no_improvement = n_checks_no_improvement
        self.max_to_keep = max_to_keep
        self.patience = patience
        self.must_reach_loss = must_reach_loss
        self.good_enough_loss = good_enough_loss

        if loss_name:
            if isinstance(loss_name, str):
                assert loss_name in ['ks_dist', 'corr', 'emd'], \
                    "Only 'ks_dist', 'corr', 'emd' supported, given loss_name='{}'".format(loss_name)
            elif isinstance(loss_name, list):
                assert np.all([l in ['ks_dist', 'corr', 'emd'] for l in loss_name]), \
                    "Only 'ks_dist', 'corr', 'emd' supported, given loss_name='{}'".format(loss_name)
        self.loss_name = loss_name

        self.count_no_improvement: int = 0
        self.best_loss: Optional[float] = None
        self.best_checkpoint: Optional[str] = None
        self.best_iteration: int = 0

        if self.use_checkpointing:
            self.checkpoint = tf.train.Checkpoint()
            self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, directory=self.checkpoint_path,
                                                                 max_to_keep=self.max_to_keep)

    def stop_learning_check_loss(self, iteration: int, loss: dict) -> bool:
        """Compare the loss against previous iteration, evaluate the criteria and return accordingly.

        Args:
            iteration: Iteration number.
            loss: OrderedDict containing all losses.

        Returns
            bool: True if criteria are met to stop learning.
        """
        if iteration % self.check_frequency != 0:
            return False

        if self.loss_name:
            if isinstance(self.loss_name, str):
                loss = {self.loss_name: loss[self.loss_name]}
            if isinstance(self.loss_name, list):
                loss = {k: v for k, v in loss.items() if k in self.loss_name}

        total_loss = np.nanmean(np.concatenate(list(loss.values())))

        if np.isnan(total_loss):
            logger.error('LearningManager :: Total Loss is NaN')
            return False

        logger.debug("LearningManager :: Iteration {}. Current Loss = {:.4f}".format(iteration, total_loss))

        if iteration < self.patience:
            return False
        if self.must_reach_loss and total_loss > self.must_reach_loss:
            return False
        if self.good_enough_loss and total_loss < self.good_enough_loss:
            return True

        if not self.best_loss or total_loss < self.best_loss:
            self.best_loss = total_loss
            self.best_iteration = iteration
            self.count_no_improvement = 0
            if self.use_checkpointing:
                current_checkpoint = self.checkpoint_manager.save()
                logger.info("LearningManager :: New loss minimum {:.4f} found at iteration {}, saving in '{}'".format(
                    total_loss, iteration, current_checkpoint))
                self.best_checkpoint = current_checkpoint
        else:
            self.count_no_improvement += 1
            if self.count_no_improvement >= self.n_checks_no_improvement:
                if self.use_checkpointing:
                    logger.info("LearningManager :: The model hasn't improved between iterations {1} and {0}. "
                                "Restoring model from iteration {1} with loss {2:.4f}".format(iteration,
                                                                                              self.best_iteration,
                                                                                              self.best_loss))
                    # Restore best model and stop learning.
                    self.checkpoint.restore(self.best_checkpoint)
                else:
                    logger.info("LearningManager :: The model hasn't improved between iterations {1} and {0}. Stopping "
                                "learning procedure")
                return True

        return False

    def stop_learning_check_data(self, iteration: int, df_orig: pd.DataFrame, df_synth: pd.DataFrame) -> bool:
        """Given original an synthetic data, calculate the loss and compare it to previous iteration, evaluate the criteria
        and return accordingly.

        Args:
            iteration: Iteration number.
            df_orig: Original DataFrame.
            df_synth: Synthesized DataFrame.

        Returns
            bool: True if criteria are met to stop learning.
        """
        if iteration % self.check_frequency != 0:
            return False

        losses = self._calculate_loss_from_data(df_orig=df_orig, df_synth=df_synth)
        return self.stop_learning_check_loss(iteration=iteration, loss=losses)

    def stop_learning(self, iteration: int, synthesizer: Synthesizer, df_train: pd.DataFrame, sample_size: int) -> bool:
        """Given a Synthesizer and the original data, get synthetic data, calculate the loss, compare it to previous
        iteration, evaluate the criteria and return accordingly.

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

        sample_size = min(sample_size, len(df_train))
        df_synth = synthesizer.synthesize(num_rows=sample_size)
        return self.stop_learning_check_data(iteration, df_train.sample(sample_size), df_synth)

    def _calculate_loss_from_data(self, df_orig: pd.DataFrame, df_synth: pd.DataFrame) -> Dict[str, List[float]]:
        """Calculate loss dictionary given two datasets. Each item in the dictionary will include a key (loss name, allowed
        options are 'ks_dist', 'corr' and 'emd'), and a value (list of losses per column).

        Args
            df_orig: Original DataFrame.
            df_synth: Synthesized DataFrame.

        Returns
            bool: True if criteria are met to stop learning.
        """
        ks_distances = []
        emd = []

        for col in df_orig.columns:
            if df_orig[col].dtype.kind in ('f', 'i'):
                ks_distances.append(ks_2samp(df_orig[col], df_synth[col])[0])
            else:
                try:
                    ks_distances.append(ks_2samp(
                        pd.to_numeric(pd.to_datetime(df_orig[col])),
                        pd.to_numeric(pd.to_datetime(df_synth[col]))
                    )[0])
                except ValueError:
                    emd.append(categorical_emd(df_orig[col].dropna(), df_synth[col].dropna()))

        corr = (df_orig.corr(method='kendall') - df_synth.corr(method='kendall')).abs().mean()

        losses = dict(
            ks_dist=list(ks_distances),
            corr=list(corr),
            emd=list(emd)
        )

        return losses
