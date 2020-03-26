import logging

import ax
from ray.tune.suggest.suggestion import SuggestionAlgorithm


logger = logging.getLogger(__name__)


class AxSearch2(SuggestionAlgorithm):
    """A wrapper around Ax to provide trial suggestions."""
    def __init__(self, ax_client, max_concurrent=10, mode="max", standard_error=0.05, **kwargs):
        assert ax is not None, "Ax must be installed!"
        assert type(max_concurrent) is int and max_concurrent > 0
        self._ax = ax_client
        exp = self._ax.experiment
        self._objective_name = exp.optimization_config.objective.metric.name
        if self._ax._enforce_sequential_optimization:
            logger.warning("Detected sequential enforcement. Setting max "
                           "concurrency to 1.")
            max_concurrent = 1
        self._max_concurrent = self._ax.generation_strategy._curr.max_parallelism
        self._parameters = list(exp.parameters)
        self._live_index_mapping = {}
        self._standard_error = standard_error

        self._num_complete = 0
        self._enforce_num_trials = self._ax.generation_strategy._curr.enforce_num_trials
        self._min_trials_observed = self._ax.generation_strategy._curr.min_trials_observed
        self._num_trials = self._ax.generation_strategy._curr.num_trials

        super(AxSearch2, self).__init__(
            metric=self._objective_name, mode=mode, **kwargs)

    def suggest(self, trial_id):
        if self._enforce_num_trials and self._num_trials <= self._counter:
            if self._num_complete < self._min_trials_observed:
                return None
            else:
                self._enforce_num_trials = self._ax.generation_strategy._curr.enforce_num_trials
                self._min_trials_observed = self._ax.generation_strategy._curr.min_trials_observed
                self._num_trials = self._ax.generation_strategy._curr.num_trials + self._num_trials

        self._max_concurrent = self._ax.generation_strategy._curr.max_parallelism
        if self._num_live_trials() >= self._max_concurrent:
            return None
        parameters, trial_index = self._ax.get_next_trial()
        self._live_index_mapping[trial_id] = trial_index
        return parameters

    def on_trial_result(self, trial_id, result):
        pass

    def on_trial_complete(self, trial_id, result=None, error=False, early_terminated=False):
        """Notification for the completion of trial.

        Data of form key value dictionary of metric names and values.
        """
        if result:
            self._num_complete += 1
            self._process_result(trial_id, result, early_terminated)
        self._live_index_mapping.pop(trial_id)

    def _process_result(self, trial_id, result, early_terminated=False):
        if early_terminated and self._use_early_stopped is False:
            return
        ax_trial_index = self._live_index_mapping[trial_id]
        metric_dict = {
            self._objective_name: (result[self._objective_name], self._standard_error * result[self._objective_name])
        }
        outcome_names = [
            oc.metric.name for oc in
            self._ax.experiment.optimization_config.outcome_constraints
        ]
        metric_dict.update(
            {on: (result[on], self._standard_error * result[self._objective_name]) for on in outcome_names})
        self._ax.complete_trial(
            trial_index=ax_trial_index, raw_data=metric_dict)

    def _num_live_trials(self):
        return len(self._live_index_mapping)

    def restore(self, checkpoint_dir):
        pass

    def save(self, checkpoint_dir):
        pass
