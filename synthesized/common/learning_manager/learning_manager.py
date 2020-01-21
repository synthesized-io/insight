from typing import Optional, Dict, List, Union
import logging

import tensorflow as tf
import pandas as pd
import numpy as np

from ...common.synthesizer import Synthesizer
from ...testing.metrics import calculate_evaluation_metrics

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
                 stop_metric_name: Union[str, List[str], None] = None, sample_size: Optional[int] = 10_000):
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
                if stop_metric_name not in allowed_stop_metric_names:
                    raise ValueError("Only {} supported, given stop_metric='{}'"
                                     .format(stop_metric_name, allowed_stop_metric_names))
            elif isinstance(stop_metric_name, list):
                if not np.all([l in allowed_stop_metric_names for l in stop_metric_name]):
                    raise ValueError("Only {} supported, given stop_metric='{}'"
                                     .format(stop_metric_name, allowed_stop_metric_names))
        self.stop_metric_name = stop_metric_name
        self.sample_size = sample_size

        self.count_no_improvement: int = 0
        self.best_stop_metric: Optional[float] = None
        self.best_checkpoint: Optional[str] = None
        self.best_iteration: int = 0

        if self.use_checkpointing:
            self.checkpoint = tf.train.Checkpoint()
            self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, directory=self.checkpoint_path,
                                                                 max_to_keep=self.max_to_keep)

    def set_check_frequency(self, batch_size: int):
        self.check_frequency = int(1e4 / np.sqrt(batch_size))

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

        stop_metrics = calculate_evaluation_metrics(df_orig=df_orig, df_synth=df_synth, column_names=column_names)
        return self.stop_learning_check_metric(iteration=iteration, stop_metric=stop_metrics)

    def stop_learning_synthesizer(self, iteration: int, synthesizer: Synthesizer, df_train_orig: pd.DataFrame) -> bool:
        """Given a Synthesizer and the original data, get synthetic data, calculate the 'stop_metric', compare it to
        previous iteration, evaluate the criteria and return accordingly.

        Args:
            iteration: Iteration number.
            synthesizer: Synthesizer object, with 'synthesize(num_rows)' method.
            df_train_orig: Original DataFrame.

        Returns
            bool: True if criteria are met to stop learning.
        """
        if iteration % self.check_frequency != 0:
            return False

        if len(synthesizer.get_conditions()) > 0:
            raise NotImplementedError

        sample_size = min(self.sample_size, len(df_train_orig)) if self.sample_size else len(df_train_orig)
        column_names = [col for v in synthesizer.get_values() for col in v.learned_input_columns()]
        if len(column_names) == 0:
            return False

        df_synth = synthesizer.synthesize(num_rows=sample_size)
        return self.stop_learning_check_data(iteration, df_train_orig.sample(sample_size), df_synth,
                                             column_names=column_names)

    def stop_learning_vae_loss(self, iteration: int, synthesizer: Synthesizer, data_dict: Dict[tf.Tensor, np.array],
                               num_data: int):
        """Given a Synthesizer and the original data, get synthetic data, calculate the VAE loss, compare it to
        previous iteration, evaluate the criteria and return accordingly.

        Args
            iteration: Iteration number.
            synthesizer: Synthesizer object, with 'synthesize(num_rows)' method.
            df_train: Preprocessed DataFrame.
            num_data: Validation batch size.

        Returns
            bool: True if criteria are met to stop learning.
        """
        if iteration % self.check_frequency != 0:
            return False

        batch_valid = np.random.randint(num_data, size=self.sample_size)
        feed_dict = {placeholder: value_data[batch_valid] for placeholder, value_data in data_dict.items()}

        loss_tensor = synthesizer.get_losses()
        losses = synthesizer.run(fetches=loss_tensor, feed_dict=feed_dict)
        losses = {k: [v] for k, v in losses.items() if k in ['reconstruction-loss', 'encoding']}
        return self.stop_learning_check_metric(iteration, losses)

    def stop_learning(self, iteration: int, synthesizer: Synthesizer, use_vae_loss: bool = True,
                      data_dict: Dict[tf.Tensor, np.array] = None, num_data: int = None,
                      df_train_orig: pd.DataFrame = None):
        """Given all the parameters, compare current iteration to previous on, evaluate the criteria and return
        accordingly.

        Args
            iteration: Iteration number.
            synthesizer: Synthesizer object, with 'synthesize(num_rows)' method.
            use_vae_loss: Whether to use VAE loss or Evaluation Metrics.
            data_dict: Dictionary containing tensors and arrays to be used in 'feed_dict' after sampling it down.
            num_data: Validation batch size.
            df_train: Preprocessed DataFrame.

        Returns
            bool: True if criteria are met to stop learning.
        """
        if use_vae_loss:
            assert data_dict is not None and num_data is not None
            return self.stop_learning_vae_loss(iteration, synthesizer=synthesizer, data_dict=data_dict,
                                               num_data=num_data)
        else:
            assert df_train_orig is not None
            return self.stop_learning_synthesizer(iteration, synthesizer=synthesizer, df_train_orig=df_train_orig)

    def restart_learning_manager(self):
        self.count_no_improvement: int = 0
        self.best_stop_metric: Optional[float] = None
        self.best_checkpoint: Optional[str] = None
        self.best_iteration: int = 0
