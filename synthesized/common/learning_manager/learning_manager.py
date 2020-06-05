import logging
import tempfile
import time
from typing import Optional, Dict, List, Union, Callable, Any

import numpy as np
import pandas as pd
import tensorflow as tf

from ..synthesizer import Synthesizer
from ...values import ValueFactory
from ...insight.evaluation import calculate_evaluation_metrics

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
                 checkpoint_path: str = None, max_training_time: float = None,
                 n_checks_no_improvement: int = 10, max_to_keep: int = 3, patience: int = 750,
                 tol: float = 1e-4, must_reach_metric: float = None, good_enough_metric: float = None,
                 stop_metric_name: Union[str, List[str], None] = None, sample_size: Optional[int] = 10_000,
                 use_vae_loss: bool = True, custom_stop_metric: Callable[[pd.DataFrame, pd.DataFrame], float] = None
                 ):
        """Initialize LearningManager.

        Args:
            check_frequency: Frequency to perform checks.
            use_checkpointing: Whether to store checkpoints each 'check_frequency'.
            checkpoint_path: Directory where checkpoints will be saved.
            max_training_time: Maximum training time.
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
            sample_size: Sample size
            use_vae_loss: Whether to use the VAE learning loss or evaluation metrics.
            custom_stop_metric: If given, use this callable to compute stop metric.
        """

        self.check_frequency = check_frequency
        self.use_checkpointing = use_checkpointing
        if checkpoint_path is not None:
            self.checkpoint_path = checkpoint_path
        else:
            self.checkpoint_path = tempfile.mkdtemp()
        self.n_checks_no_improvement = n_checks_no_improvement
        self.max_to_keep = max_to_keep
        self.patience = patience
        self.tol = tol
        self.must_reach_metric = must_reach_metric
        self.good_enough_metric = good_enough_metric

        allowed_stop_metric_names = ['ks_distance', 'corr_dist', 'emd_categ']
        if stop_metric_name:
            if isinstance(stop_metric_name, str):
                if stop_metric_name not in allowed_stop_metric_names:
                    raise ValueError("Only {} supported, given stop_metric='{}'"
                                     .format(stop_metric_name, allowed_stop_metric_names))
            elif isinstance(stop_metric_name, list):
                if not np.all([metric_name in allowed_stop_metric_names for metric_name in stop_metric_name]):
                    raise ValueError("Only {} supported, given stop_metric='{}'"
                                     .format(stop_metric_name, allowed_stop_metric_names))
        self.stop_metric_name = stop_metric_name
        self.sample_size = sample_size
        self.max_training_time = max_training_time
        self.use_vae_loss = use_vae_loss
        self.custom_stop_metric = custom_stop_metric

        self.count_no_improvement: int = 0
        self.best_stop_metric: Optional[float] = None
        self.best_checkpoint: Optional[str] = None
        self.best_iteration: int = 0
        self.start_time: Optional[float] = None

        if self.use_checkpointing:
            self.checkpoint = tf.train.Checkpoint()
            self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, directory=self.checkpoint_path,
                                                                 max_to_keep=self.max_to_keep)

    def set_check_frequency(self, batch_size: int):
        self.check_frequency = int(1e3 / np.sqrt(batch_size))
        logger.debug("LearningManager :: check_frequency updated to {}".format(self.check_frequency))

    def stop_learning_check_metric(self, iteration: int, stop_metric: Union[Dict[str, List[float]], float]
                                   ) -> bool:
        """Compare the 'stop_metric' against previous iteration, evaluate the criteria and return accordingly.

        Args:
            iteration: Iteration number.
            stop_metric: OrderedDict containing all stop_metrics, or float.

        Returns
            bool: True if criteria are met to stop learning.
        """
        if iteration % self.check_frequency != 0:
            return False

        if isinstance(stop_metric, dict):
            if self.stop_metric_name:
                if isinstance(self.stop_metric_name, str):
                    stop_metric = {self.stop_metric_name: stop_metric[self.stop_metric_name]}
                if isinstance(self.stop_metric_name, list):
                    stop_metric = {k: v for k, v in stop_metric.items() if k in self.stop_metric_name}

            total_stop_metric = np.nanmean(np.concatenate(list(stop_metric.values())))

        elif isinstance(stop_metric, float):
            total_stop_metric = stop_metric

        else:
            raise TypeError("Given 'stop_metric' type not supported.")

        if np.isnan(total_stop_metric):
            logger.error("LearningManager :: Total 'stop_metric' is NaN")
            return False

        if self.max_training_time is not None and self.start_time is not None:
            if time.time() - self.start_time > self.max_training_time:
                if self.use_checkpointing:
                    if not self.best_stop_metric or total_stop_metric < self.best_stop_metric:
                        self.best_stop_metric = total_stop_metric
                        self.best_iteration = iteration
                        self.count_no_improvement = 0
                        current_checkpoint = self.checkpoint_manager.save()
                        self.best_checkpoint = current_checkpoint
                        logger.info(
                            "LearningManager :: The model has reached the maximum training time ({2:.2f}s). "
                            "and lowest stop_metric={1:.4f} at iteration {0}".format(
                                self.best_iteration, self.best_stop_metric or 0, self.max_training_time
                            )
                        )
                        return True

                    if self.best_checkpoint is not None:
                        logger.info(
                            "LearningManager :: The model has reached the maximum training time ({2:.2f}s). "
                            "Restoring model from iteration {0} with stop_metric={1:.4f}".format(
                                self.best_iteration, self.best_stop_metric or 0, self.max_training_time
                            )
                        )
                        # Restore best model and stop learning.
                        self.checkpoint.restore(self.best_checkpoint)
                        return True

                else:
                    logger.info("LearningManager :: The model has reached maximum training time ({0:.2f}s). Stopping "
                                "learning procedure".format(self.max_training_time))
                    return True

        logger.debug("LearningManager :: Iteration {}. Current stop_metric={:.4f}".format(iteration, total_stop_metric))

        if iteration < self.patience:
            return False
        if self.must_reach_metric and total_stop_metric > self.must_reach_metric:
            return False
        if self.good_enough_metric and total_stop_metric < self.good_enough_metric:
            return True

        if not self.best_stop_metric or total_stop_metric < self.best_stop_metric - self.tol:
            logger.debug("LearningManager :: New stop_metric minimum {:.4f} found at iteration {} after {}/{} checks "
                         "without improvement"
                         .format(total_stop_metric, iteration, self.count_no_improvement, self.n_checks_no_improvement))
            self.best_stop_metric = total_stop_metric
            self.best_iteration = iteration
            self.count_no_improvement = 0
            if self.use_checkpointing:
                current_checkpoint = self.checkpoint_manager.save()
                self.best_checkpoint = current_checkpoint
        else:
            self.count_no_improvement += 1
            if self.count_no_improvement >= self.n_checks_no_improvement:
                if self.use_checkpointing:
                    logger.info("LearningManager :: The model hasn't improved between iterations {1} and {0}. Restoring"
                                " model from iteration {1} with stop_metric={2:.4f}".format(iteration,
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
                                 value_factory: ValueFactory, column_names: Optional[List[str]] = None) -> bool:
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

        if self.custom_stop_metric is None:
            stop_metrics: Union[Dict[str, Union[pd.Series, pd.DataFrame]], float] = calculate_evaluation_metrics(
                df_orig=df_orig, df_synth=df_synth, value_factory=value_factory, column_names=column_names
            )
        else:
            stop_metrics = self.custom_stop_metric(df_orig, df_synth)

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
                                             column_names=column_names, value_factory=synthesizer.value_factory)

    def stop_learning_vae_loss(self, iteration: int, synthesizer: Synthesizer, data_dict: Dict[str, tf.Tensor]) -> bool:
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

        batch_valid = tf.random.uniform(shape=(self.sample_size,), maxval=list(data_dict.values())[0].shape[0],
                                        dtype=tf.int64)
        feed_dict = {name: tf.nn.embedding_lookup(params=value_data, ids=batch_valid)
                     for name, value_data in data_dict.items()}

        losses = synthesizer.get_losses(data=feed_dict)
        losses = {k: [v] for k, v in losses.items() if k in ['reconstruction-loss', 'kl-loss']}
        return self.stop_learning_check_metric(iteration, losses)

    def stop_learning(self, iteration: int, synthesizer: Synthesizer,
                      data_dict: Dict[str, tf.Tensor] = None, num_data: int = None,
                      df_train_orig: pd.DataFrame = None
                      ) -> bool:
        """Given all the parameters, compare current iteration to previous on, evaluate the criteria and return
        accordingly.

        Args
            iteration: Iteration number.
            synthesizer: Synthesizer object, with 'synthesize(num_rows)' method.
            data_dict: Dictionary containing tensors and arrays to be used in 'feed_dict' after sampling it down.
            num_data: Validation batch size.
            df_train: Preprocessed DataFrame.

        Returns
            bool: True if criteria are met to stop learning.
        """
        if iteration % self.check_frequency != 0:
            return False

        if self.use_vae_loss:
            assert data_dict is not None and num_data is not None
            return self.stop_learning_vae_loss(iteration, synthesizer=synthesizer, data_dict=data_dict)
        else:
            assert df_train_orig is not None
            return self.stop_learning_synthesizer(iteration, synthesizer=synthesizer, df_train_orig=df_train_orig)

    def restart_learning_manager(self):
        self.count_no_improvement: int = 0
        self.best_stop_metric: Optional[float] = None
        self.best_checkpoint: Optional[str] = None
        self.best_iteration: int = 0
        self.start_time: float = time.time()

    def get_variables(self) -> Dict[str, Any]:
        return dict(
            max_training_time=self.max_training_time,
            use_vae_loss=self.use_vae_loss,
            custom_stop_metric=self.custom_stop_metric
        )

    def set_variables(self, variables: Dict[str, Any]):
        self.max_training_time = variables['max_training_time']
        self.use_vae_loss = variables['use_vae_loss']
        self.custom_stop_metric = variables['custom_stop_metric']
