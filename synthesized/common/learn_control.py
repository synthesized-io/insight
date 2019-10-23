from typing import Optional, Dict

import tensorflow as tf
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt


class LearnControl:
    """
    This class will control the learning, checking that it improves and that stopping learning if necessary before the
    maximum number of iterations is reached.
    """
    def __init__(self, check_frequency: int = 100, checkpoint_path: Optional[str] = None, max_to_keep: int = 10,
                 patience: int = 750, must_reach_loss: float = None, good_enough_loss: float = None):
        """
        Initialize LearnControl.

        :param check_frequency:
        :param checkpoint_path: Directory where checkpoints will be saved
        :param max_to_keep: If the LearnControl checks performance for 'max_to_keep' times without improvement,
            will return True.
        :param patience: How many iterations before start checking the performance.
        :param must_reach_loss: If this loss threshold is not reached, will always return False.
        :param good_enough_loss: If this loss threshold is not reached, will return True even if the model is still
            improving.
        """

        self.check_frequency = check_frequency
        self.checkpoint_path = checkpoint_path if checkpoint_path is not None else '/tmp/tf_checkpoints'
        self.max_to_keep = max_to_keep
        self.patience = patience
        self.must_reach_loss = must_reach_loss
        self.good_enough_loss = good_enough_loss

        self.loss_log: Dict[int, dict] = dict()

        self.count_no_improvement = 0
        self.best_loss: Optional[float] = None
        self.best_checkpoint = None
        self.best_iteration = 0

        self.checkpoint = tf.train.Checkpoint()
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, directory=self.checkpoint_path,
                                                             max_to_keep=self.max_to_keep)

    def checkpoint_model_from_loss(self, iteration: int, loss: dict) -> bool:
        """
        Compare the loss against previous iteration, evaluate the criteria and return accordingly

        :param iteration:
        :param loss: OrderedDict containing all losses.
        :return: Returns True if criteria are met to stop learning.
        """

        self.loss_log[iteration] = loss

        total_loss = sum(loss.values())
        # print("LearnControl :: Iteration {}. Current Loss = {:.4f}".format(iteration, total_loss))

        if iteration < self.patience:
            return False
        if self.must_reach_loss and total_loss > self.must_reach_loss:
            return False
        if self.good_enough_loss and total_loss < self.good_enough_loss:
            return True

        if not self.best_loss or total_loss < self.best_loss:
            current_checkpoint = self.checkpoint_manager.save()
            print("LearnControl :: New loss minimum {:.4f} found at iteration {}, saving in '{}'"
                  .format(total_loss, iteration, current_checkpoint))
            self.best_loss = total_loss
            self.best_checkpoint = current_checkpoint
            self.best_iteration = iteration
            self.count_no_improvement = 0
        else:
            self.count_no_improvement += 1
            if self.count_no_improvement >= self.max_to_keep:
                print("LearnControl :: The model hasn't improved between iterations {1} and {0}. Restoring model from "
                      "iteration {1} with loss {2:.4f}".format(iteration, self.best_iteration, self.best_loss))
                self.restore_best_model()
                return True
        return False

    def checkpoint_model_from_data(self, iteration: int, df_orig: pd.DataFrame, df_synth: pd.DataFrame) -> bool:
        loss = self.calculate_loss_from_data(df_orig, df_synth)
        return self.checkpoint_model_from_loss(iteration=iteration, loss=loss)

    def calculate_loss_from_data(self, df_orig: pd.DataFrame, df_synth: pd.DataFrame) -> dict:
        ks_distance = []
        for col in df_orig.columns.values:
            try:
                ks_distance.append(ks_2samp(df_orig[col], df_synth[col])[0])
            except Exception as e:
                print('WARNING :: {}'.format(e))

        corr = 0
        try:
            corr = (df_orig.corr() - df_synth.corr()).abs().mean()
        except Exception as e:
            print('WARNING :: {}'.format(e))

        return dict(
            corr_avg=np.mean(corr),
            ks_dist_avg=np.mean(ks_distance)
        )

    def restore_best_model(self) -> bool:
        return self.checkpoint.restore(self.best_checkpoint)

    def plot_learning(self):
        t = list(self.loss_log.keys())
        x = []
        for v in self.loss_log.values():
            x.append(list(v.values()))
        labels = list(v.keys())
        x = np.array(x).T

        plt.figure(figsize=(12, 8))
        for i in range(len(x)):
            plt.plot(t, x[i], label=labels[i])
        x_avg = x.mean(axis=0)
        plt.plot(t, x_avg, 'k--', label='Average')
        plt.plot([min(t), max(t)], [min(x_avg), min(x_avg)], 'k:', label='Minimum')
        plt.legend()
        plt.xlabel('Iteration')
        plt.show()
