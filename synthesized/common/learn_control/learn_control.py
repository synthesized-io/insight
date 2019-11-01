from typing import Optional, Dict, List, Union
import logging

import tensorflow as tf
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt

from ..util import categorical_emd
from ...synthesizer import Synthesizer

logger = logging.getLogger(__name__)


class LearnControl:
    """
    This class will control the learning, checking that it improves and that stopping learning if necessary before the
    maximum number of iterations is reached.

    Usage example:

        > lc = LearnControl()
        > for iteration in range(iterations):
        >     batch = get_batch()
        >     synthesizer.learn(batch)
        >     if lc.checkpoint_model_from_synthesizer(iteration, synthesizer=synthesizer, df_train=df_train):
        >         break


    """
    def __init__(self, check_frequency: int = 100, checkpoint_path: Optional[str] = None,
                 no_learn_iterations: int = 10, max_to_keep: int = 5,
                 patience: int = 750, must_reach_loss: float = None, good_enough_loss: float = None,
                 which_loss: Optional[Union[str, List[str]]] = None):
        """
        Initialize LearnControl.

        :param check_frequency:
        :param checkpoint_path: Directory where checkpoints will be saved
        :param no_learn_iterations: If the LearnControl checks performance for 'no_learn_iterations' times without
            improvement, will return True and stop learning.
        :param max_to_keep: Defines the number of previous checkpoints stored.
        :param patience: How many iterations before start checking the performance.
        :param must_reach_loss: If this loss threshold is not reached, will always return False.
        :param good_enough_loss: If this loss threshold is not reached, will return True even if the model is still
            improving.
        :param which_loss: Which loss will be evaluated in each iteration. If None, all losses will be used, otherwise
            a str for a single loss or a list of str for more than one.
        """

        self.check_frequency = check_frequency
        self.checkpoint_path = checkpoint_path if checkpoint_path is not None else '/tmp/tf_checkpoints'
        self.no_learn_iterations = no_learn_iterations
        self.max_to_keep = max_to_keep
        self.patience = patience
        self.must_reach_loss = must_reach_loss
        self.good_enough_loss = good_enough_loss

        if which_loss:
            assert which_loss in ['ks_dist_avg', 'corr_avg', 'emd_avg'], \
                "Only 'ks_dist_avg', 'corr_avg', 'emd_avg' supported, given which_loss='{}'".format(which_loss)
        self.which_loss = which_loss

        self.loss_log: Dict[int, dict] = dict()

        self.count_no_improvement: int = 0
        self.best_loss: Optional[float] = None
        self.best_checkpoint: Optional[str] = None
        self.best_iteration: int = 0

        self.checkpoint = tf.train.Checkpoint()
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, directory=self.checkpoint_path,
                                                             max_to_keep=self.max_to_keep)

        # Remove this:
        self.best_lc_loss: Optional[float] = None
        self.best_lc_iteration: Optional[int] = None
        self.test_lc: bool = False

    def checkpoint_model_from_loss(self, iteration: int, loss: dict) -> bool:
        """
        Compare the loss against previous iteration, evaluate the criteria and return accordingly.

        :param iteration:
        :param loss: OrderedDict containing all losses.
        :return: Returns True if criteria are met to stop learning.
        """

        self.loss_log[iteration] = loss

        if self.which_loss:
            if isinstance(self.which_loss, int):
                loss = loss[self.which_loss]
            if isinstance(self.which_loss, list):
                loss = {k: v for k, v in loss.items() if k in self.which_loss}

        if len(loss) == 1:
            total_loss = np.nanmean(list(loss.values()))
        else:
            total_loss = np.nanmean(np.concatenate(list(loss.values())))

        if np.isnan(total_loss):
            logger.error('LearnControl :: Total Loss is NaN')
            return False

        logger.debug("LearnControl :: Iteration {}. Current Loss = {:.4f}".format(iteration, total_loss))

        if iteration < self.patience:
            return False
        if self.must_reach_loss and total_loss > self.must_reach_loss:
            return False
        if self.good_enough_loss and total_loss < self.good_enough_loss:
            return True

        if not self.best_loss or total_loss < self.best_loss:
            current_checkpoint = self.checkpoint_manager.save()
            logger.info("LearnControl :: New loss minimum {:.4f} found at iteration {}, saving in '{}'".format(
                total_loss, iteration, current_checkpoint))
            self.best_loss = total_loss
            self.best_checkpoint = current_checkpoint
            self.best_iteration = iteration
            self.count_no_improvement = 0
        else:
            self.count_no_improvement += 1
            if self.count_no_improvement >= self.no_learn_iterations:
                logger.info("LearnControl :: The model hasn't improved between iterations {1} and {0}. Restoring model "
                            "from iteration {1} with loss {2:.4f}".format(iteration, self.best_iteration,
                                                                          self.best_loss))

                if not self.test_lc:
                    self.restore_best_model()
                    return True
                else:
                    # Remove this:
                    if not self.best_lc_loss:
                        self.best_lc_loss = self.best_loss
                        self.best_lc_iteration = self.best_iteration
                    return True

        return False

    def checkpoint_model_from_data(self, iteration: int, df_orig: pd.DataFrame, df_synth: pd.DataFrame) -> bool:
        """
        Given original an synthetic data, calculate the loss and cmopare it to previous iteration, evaluate the criteria
        and return accordingly.

        :param iteration:
        :param df_orig:
        :param df_synth:
        :return:
        """
        losses = self.calculate_loss_from_data(df_orig=df_orig, df_synth=df_synth)
        return self.checkpoint_model_from_loss(iteration=iteration, loss=losses)

    def checkpoint_model_from_synthesizer(self, iteration: int, synthesizer: Synthesizer, df_train: pd.DataFrame,
                                          sample_size: int):
        """
        Given a Synthesizer and the original data, get synthetic data, calculate the loss, compare it to previous
        iteration, evaluate the criteria and return accordingly.

        :param iteration:
        :param synthesizer:
        :param df_train:
        :param sample_size:
        :return:
        """
        if iteration % self.check_frequency != 0:
            return False

        sample_size = min(sample_size, len(df_train))
        df_synth = synthesizer.synthesize(num_rows=sample_size)
        return self.checkpoint_model_from_data(iteration, df_train.sample(sample_size), df_synth)

    def calculate_loss_from_data(self, df_orig: pd.DataFrame, df_synth: pd.DataFrame) -> dict:
        """
        Calculate loss dictionary given two datasets

        :param df_orig:
        :param df_synth:
        :return:
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
            ks_dist_avg=list(ks_distances),
            corr_avg=list(corr),
            emd_avg=list(emd)
        )

        return losses

    def restore_best_model(self) -> bool:
        return self.checkpoint.restore(self.best_checkpoint)

    def plot_learning(self, fig_name=None):
        if len(self.loss_log) < 2:
            return

        t = list(self.loss_log.keys())
        x = []
        for v in self.loss_log.values():
            x.append([np.nanmean(l) for l in v.values()])
        labels = list(v.keys())
        x = np.array(x).T

        plt.figure(figsize=(12, 8))
        for i in range(len(x)):
            plt.plot(t, x[i], label=labels[i])
        x_avg = x.mean(axis=0)
        plt.plot(t, x_avg, 'k--', label='Average')

        x_min = min(x_avg)
        t_min = t[int(np.argmin(x_avg))]
        plt.plot([min(t), max(t)], [x_min, x_min], 'k:', label='Minimum')
        plt.plot([t_min, t_min], [np.min(x), np.max(x)], 'k:')

        # Remove this:
        if self.test_lc:
            plt.plot([min(t), max(t)], [self.best_lc_loss, self.best_lc_loss], 'k-.', label='Minimum LC')
            plt.plot([self.best_lc_iteration, self.best_lc_iteration], [np.min(x), np.max(x)], 'k-.')

        plt.legend()
        plt.xlabel('Iteration')
        if fig_name:
            plt.savefig('{}.png'.format(fig_name))
        plt.show()

        return x_min, t_min

    def empty_mean(self, l):
        return np.nanmean(l) if len(l) > 0 else 0.
