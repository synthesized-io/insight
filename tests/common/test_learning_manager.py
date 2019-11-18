from typing import Optional, Union, List, Callable, Dict

from synthesized.common.learning_manager import LearningManager


def _test_learning_manager(
        fn_metric: Callable[[int], Dict[str, float]], iterations: int, expected_iteration_break: Optional[int],
        # Learning Manager params:
        check_frequency: int = 100, use_checkpointing: bool = True,
        checkpoint_path: str = '/tmp/tf_checkpoints',
        n_checks_no_improvement: int = 10, max_to_keep: int = 3,
        patience: int = 750, must_reach_metric: float = None, good_enough_metric: float = None,
        stop_metric_name: Optional[Union[str, List[str]]] = None):
    lm = LearningManager(
        check_frequency=check_frequency,
        use_checkpointing=use_checkpointing,
        checkpoint_path=checkpoint_path,
        n_checks_no_improvement=n_checks_no_improvement,
        max_to_keep=max_to_keep,
        patience=patience,
        must_reach_metric=must_reach_metric,
        good_enough_metric=good_enough_metric,
        stop_metric_name=stop_metric_name
    )

    iteration_break = None
    for i in range(iterations):
        loss = fn_metric(i)
        if lm.stop_learning_check_metric(i, loss):
            iteration_break = i
            break

    assert (iteration_break == expected_iteration_break) or (expected_iteration_break is None and i == iterations - 1)
    return


def test_lm_basic():
    iterations = 2500

    def fn_metric(i):
        return {k: [(iterations - i) / float(iterations)] for k in ['ks_dist', 'corr', 'emd']}

    _test_learning_manager(fn_metric, iterations, expected_iteration_break=None, use_checkpointing=False)


def test_lm_basic2():
    iterations = 2500

    def fn_metric(i):
        if i <= 1000:
            return {k: [(iterations - i) / float(iterations)] for k in ['ks_dist', 'corr', 'emd']}
        else:
            return {k: [(iterations + i) / float(iterations)] for k in ['ks_dist', 'corr', 'emd']}

    _test_learning_manager(fn_metric, iterations, expected_iteration_break=2000, use_checkpointing=False)


def test_lm_metric_name():
    iterations = 3000

    def fn_metric(i):
        loss = {k: [(iterations + i) / float(iterations)] for k in ['corr', 'emd']}
        if i <= 1500:
            loss['ks_dist'] = [(iterations - i) / float(iterations)]
        else:
            loss['ks_dist'] = [(iterations + i) / float(iterations)]
        return loss

    _test_learning_manager(fn_metric, iterations, expected_iteration_break=2500, use_checkpointing=False,
                           stop_metric_name='ks_dist')


def test_lm_patience():
    iterations = 2500

    def fn_metric(i):
        if i <= 1000:
            return {k: [(iterations + i) / float(iterations)] for k in ['ks_dist', 'corr', 'emd']}
        else:
            return {k: [(iterations - i) / float(iterations)] for k in ['ks_dist', 'corr', 'emd']}

    _test_learning_manager(fn_metric, iterations, expected_iteration_break=None, use_checkpointing=False)


def test_lm_4():
    iterations = 5000

    def fn_metric(i):
        if i <= 1000:
            return {k: [(iterations - i) / float(iterations)] for k in ['ks_dist', 'corr', 'emd']}
        if i <= 1500:
            return {k: [(iterations + i) / float(iterations)] for k in ['ks_dist', 'corr', 'emd']}
        if i <= 2500:
            return {k: [(iterations - i) / float(iterations)] for k in ['ks_dist', 'corr', 'emd']}
        else:
            return {k: [(iterations + i) / float(iterations)] for k in ['ks_dist', 'corr', 'emd']}

    _test_learning_manager(fn_metric, iterations, expected_iteration_break=3500, use_checkpointing=False)
