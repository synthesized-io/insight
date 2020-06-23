import datetime
from collections import OrderedDict
import time
from typing import Optional, List

try:
    from IPython.display import Markdown, display
except ImportError:
    Markdown = str
    display = print
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import simplejson
from sklearn.model_selection import train_test_split

from .plotting import plot_time_series, plot_data, plot_multidimensional
from .utility import UtilityTesting, MAX_PVAL
from .utility_time_series import TimeSeriesUtilityTesting
from ..complex.highdim import HighDimSynthesizer, HighDimConfig
from ..complex.series import SeriesSynthesizer, SeriesConfig
from ..insight import metrics
from ..metadata import MetaExtractor


class Evaluation:
    def __init__(self, branch: str, revision: str, group: str, metrics_file: str):

        self.branch = branch
        self.revision = revision
        self.group = group
        self.metrics_file = metrics_file

        self.metrics: OrderedDict = OrderedDict()
        self.configs: OrderedDict = OrderedDict()

    def record_config(self, evaluation: str, config: dict):
        self.configs[evaluation] = config

    def record_metric(self, evaluation: str, key: str, value: float):
        if evaluation not in self.metrics:
            self.metrics[evaluation] = OrderedDict()
        evaluation_metrics = self.metrics[evaluation]
        if key in evaluation_metrics:
            evaluation_metrics[key].append(value)
        else:
            evaluation_metrics[key] = [value]

    def write_metrics(self, append=True):
        timestamp = datetime.datetime.now().isoformat()
        write_mode = 'a' if append else 'w'
        with open(self.metrics_file, write_mode) as f:
            for evaluation, evaluation_metrics in self.metrics.items():
                data = OrderedDict()
                data['group'] = self.group
                data['evaluation'] = evaluation
                data['branch'] = self.branch
                data['revision'] = self.revision
                data['timestamp'] = timestamp
                if evaluation in self.configs:
                    data['config'] = simplejson.dumps(self.configs[evaluation], separators=(',', ':'), ignore_nan=True)
                for name, vals in evaluation_metrics.items():
                    data[name + '_mean'] = np.mean(vals)
                    data[name + '_std'] = np.std(vals)
                    data[name + '_count'] = len(vals)
                simplejson.dump(data, f, separators=(',', ':'), ignore_nan=True)
                f.write('\n')


# -- training functions
def synthesize_and_plot(
        data: pd.DataFrame, name: str, evaluation, config, eval_metrics: List[metrics.DataFrameComparison] = None,
        test_data: Optional[pd.DataFrame] = None, plot_basic: bool = True, plot_losses: bool = False,
        plot_distances: bool = False, show_distributions: bool = False, show_distribution_distances: bool = False,
        show_emd_distances: bool = False, show_correlation_distances: bool = False,
        show_correlation_matrix: bool = False, show_cramers_v_distances: bool = False,
        show_cramers_v_matrix: bool = False, show_logistic_rsquared_distances: bool = False,
        show_logistic_rsquared_matrix: bool = False
):
    """
    Synthesize and plot data from a Synthesizer trained on the dataframe `data`.
    """
    eval_data = test_data if test_data is not None else data
    len_eval_data = len(eval_data)
    eval_metrics = eval_metrics or []

    def callback(synth, iteration, losses):
        if len(losses) > 0 and hasattr(list(losses.values())[0], 'numpy'):
            if len(synth.loss_history) == 0:
                synth.loss_history.append({n: l.numpy() for n, l in losses.items()})
            else:
                synth.loss_history.append({local_name: losses[local_name].numpy()
                                           for local_name in synth.loss_history[0]})
        return False

    evaluation.record_config(evaluation=name, config=config)
    start = time.time()

    df_meta = MetaExtractor.extract(df=data)
    hd_config = HighDimConfig(**config['params'])
    with HighDimSynthesizer(df_meta=df_meta, config=hd_config) as synthesizer:
        synthesizer.learn(df_train=data, num_iterations=config['num_iterations'], callback=callback,
                          callback_freq=100)
        training_time = time.time() - start
        synthesized = synthesizer.synthesize(num_rows=len_eval_data)
        # value_factory = synthesizer.value_factory
        print('took', training_time, 's')

    evaluation.record_metric(evaluation=name, key='training_time', value=training_time)

    print("Metrics:")
    for metric in eval_metrics:
        value = metric(df_old=data, df_new=synthesized)
        evaluation.record_metric(evaluation=name, key=metric.name, value=value)
        print(f"{metric.name}: {value}")

    if plot_basic:
        if data.shape[1] <= 3:
            display(Markdown("## Plot data"))
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), sharex=True, sharey=True)
            ax1.set_title('orig')
            ax2.set_title('synth')
            plot_data(data, ax=ax1)
            plot_data(synthesized, ax=ax2)
            plt.show()
        else:
            display(Markdown("## Plot data"))
            fig, ax = plt.subplots(figsize=(15, 5))
            plot_multidimensional(original=data, synthetic=synthesized, ax=ax)
            plt.show()

    testing = UtilityTesting(synthesizer, data, eval_data, synthesized)

    if plot_losses:
        display(Markdown("## Show loss history"))
        df_losses = pd.DataFrame.from_records(synthesizer.loss_history)
        if len(df_losses) > 0:
            df_losses.plot(figsize=(15, 7))
            plt.show()

    if plot_distances:
        display(Markdown("## Show average distances"))
        results = testing.show_standard_metrics()

        print_line = ''
        for k, v in results.items():
            if evaluation:
                assert name, 'If evaluation is given, evaluation_name must be given too.'
                evaluation.record_metric(evaluation=name, key=k, value=v)
            print_line += '\n\t{}={:.4f}'.format(k, v)
        print(print_line)

    if show_distributions:
        display(Markdown("## Show distributions"))
        testing.show_distributions(remove_outliers=0.01)

    # First order metrics
    if show_distribution_distances:
        display(Markdown("## Show distribution distances"))
        testing.show_first_order_metric_distances(metrics.kolmogorov_smirnov_distance)
    if show_emd_distances:
        display(Markdown("## Show EMD distances"))
        testing.show_first_order_metric_distances(metrics.earth_movers_distance)

    # Second order metrics
    if show_correlation_distances:
        display(Markdown("## Show correlation distances"))
        testing.show_second_order_metric_distances(metrics.diff_kendell_tau_correlation, max_p_value=MAX_PVAL)
    if show_correlation_matrix:
        display(Markdown("## Show correlation matrices"))
        testing.show_second_order_metric_matrices(metrics.kendell_tau_correlation)
    if show_cramers_v_distances:
        display(Markdown("## Show Cramer's V distances"))
        testing.show_second_order_metric_distances(metrics.diff_cramers_v)
    if show_cramers_v_matrix:
        display(Markdown("## Show Cramer's V matrices"))
        testing.show_second_order_metric_matrices(metrics.cramers_v)
    if show_logistic_rsquared_distances:
        display(Markdown("## Show Logistic R^2 distances"))
        testing.show_second_order_metric_distances(metrics.diff_categorical_logistic_correlation,
                                                   continuous_input_only=True, categorical_output_only=True)
    if show_logistic_rsquared_matrix:
        display(Markdown("## Show Logistic R^2 matrices"))
        testing.show_second_order_metric_matrices(metrics.categorical_logistic_correlation,
                                                  continuous_input_only=True, categorical_output_only=True)

    return testing


# -- training functions
def synthesize_and_plot_time_series(
        data: pd.DataFrame, name: str, evaluation, config, eval_metrics: List[metrics.DataFrameComparison],
        test_data: Optional[pd.DataFrame] = None, col: str = "x", max_lag: int = 10, plot_basic: bool = True,
        plot_losses: bool = False, plot_distances: bool = False, show_distributions: bool = False,
        show_distribution_distances: bool = False, show_emd_distances: bool = False,
        show_correlation_distances: bool = False,
        show_correlation_matrix: bool = False, show_cramers_v_distances: bool = False,
        show_cramers_v_matrix: bool = False, show_cat_rsquared: bool = False, show_acf_distances: bool = False,
        show_pacf_distances: bool = False, show_transition_distances: bool = False, show_series: bool = False,
        show_acf=False
):
    """
    Synthesize and plot data from a Synthesizer trained on the dataframe `data`.
    """
    # len_eval_data = len(eval_data)

    def callback(synth, iteration, losses):
        if len(losses) > 0 and hasattr(list(losses.values())[0], 'numpy'):
            if len(synth.loss_history) == 0:
                synth.loss_history.append({n: l.numpy() for n, l in losses.items()})
            else:
                synth.loss_history.append({local_name: losses[local_name].numpy()
                                           for local_name in synth.loss_history[0]})
        return False

    evaluation.record_config(evaluation=name, config=config)
    start = time.time()
    identifier_label = None

    if 'identifier_label' in config['params'].keys() and config['params']['identifier_label'] is not None:
        identifier_label = config['params']['identifier_label']
    #     num_series = eval_data[identifier_label].nunique()
    #     series_length = int(len_eval_data / num_series)
    # else:
    #     num_series = 1
    #     series_length = len_eval_data

    df_meta = MetaExtractor.extract(df=data)
    series_config = SeriesConfig(**config['params'])

    with SeriesSynthesizer(df_meta=df_meta, config=series_config) as synthesizer:
        # synthesizer.learn(df_train=data, num_iterations=config['num_iterations'], callback=callback,
        #                   callback_freq=100)
        training_time = time.time() - start

        # synthesized = synthesizer.synthesize(num_series=num_series, series_length=series_length)
        synthesized = data.copy()
        identifiers = data[identifier_label].unique()
        id_map = {a: b
                  for a, b in zip(identifiers, np.random.choice(identifiers, size=len(identifiers), replace=False))}
        synthesized[identifier_label] = synthesized[identifier_label].map(id_map)

        # value_factory = synthesizer.value_factory
        print('took', training_time, 's')

    evaluation.record_metric(evaluation=name, key='training_time', value=training_time)

    print("Metrics:")
    for metric in eval_metrics:
        value = metric(df_old=data, df_new=synthesized)
        evaluation.record_metric(evaluation=name, key=metric.name, value=value)
        print(f"{metric.name}: {value}")

    if plot_basic:
        display(Markdown("## Plot time-series data"))
        fig, axes = plt.subplots(2, 2, figsize=(15, 5), sharey="row")
        fig.tight_layout()
        # original_auto_assoc = calculate_auto_association(dataset=data, col=col, max_order=max_lag)
        # synthetic_auto_assoc = calculate_auto_association(dataset=synthesized, col=col, max_order=max_lag)
        t_orig = np.arange(0, data.shape[0])
        plot_time_series(x=data[col].to_numpy(), t=t_orig, ax=axes[0, 0])
        axes[0, 0].set_title("Original")
        t_synth = np.arange(0, synthesized.shape[0])
        plot_time_series(x=synthesized[col].to_numpy(), t=t_synth, ax=axes[0, 1])
        axes[0, 1].set_title("Synthetic")

    testing = TimeSeriesUtilityTesting(synthesizer, data, synthesized, identifier=identifier_label,
                                       time_index='date')

    if plot_losses:
        # display(Markdown("## Show loss history")
        print("## Show loss history")
        df_losses = pd.DataFrame.from_records(synthesizer.loss_history)
        if len(df_losses) > 0:
            df_losses.plot(figsize=(15, 7))

    # if plot_distances:
    #     display(Markdown("## Show average distances"))
    #     results = testing.show_standard_metrics()
    #
    #     print_line = ''
    #     for k, v in results.items():
    #         if evaluation:
    #             assert name, 'If evaluation is given, evaluation_name must be given too.'
    #             evaluation.record_metric(evaluation=name, key=k, value=v)
    #         print_line += '\n\t{}={:.4f}'.format(k, v)
    #     print(print_line)

    # if show_distributions:
    #     display(Markdown("## Show distributions"))
    #     testing.show_distributions(remove_outliers=0.01)

    # # First order metrics
    # if show_distribution_distances:
    #     display(Markdown("## Show distribution distances"))
    #     testing.show_first_order_metric_distances(metrics.kolmogorov_smirnov_distance)
    # if show_emd_distances:
    #     display(Markdown("## Show EMD distances"))
    #     testing.show_first_order_metric_distances(metrics.earth_movers_distance)
    #
    # # Second order metrics
    # if show_correlation_distances:
    #     display(Markdown("## Show correlation distances"))
    #     testing.show_second_order_metric_distances(metrics.diff_kendell_tau_correlation, max_p_value=MAX_PVAL)
    # if show_correlation_matrix:
    #     display(Markdown("## Show correlation matrices"))
    #     testing.show_second_order_metric_matrices(metrics.kendell_tau_correlation)
    # if show_cramers_v_distances:
    #     display(Markdown("## Show Cramer's V distances"))
    #     testing.show_second_order_metric_distances(metrics.diff_cramers_v)
    # if show_cramers_v_matrix:
    #     display(Markdown("## Show Cramer's V matrices"), Markdown("## Show Cramer's V matrices"))
    #     testing.show_second_order_metric_matrices(metrics.cramers_v)
    # if show_cat_rsquared:
    #     display(Markdown("## Show categorical R^2"))
    #     testing.show_second_order_metric_matrices(metrics.categorical_logistic_correlation,
    #                                               continuous_inputs_only=True)

    # TIME SERIES
    if show_acf_distances:
        # display(Markdown("## Show Auto-correaltion Distances")
        print("## Show Auto-correaltion Distances")
        acf_dist_max, acf_dist_avg = testing.show_autocorrelation_distances()
        evaluation.record_metric(evaluation=name, key='acf_dist_max', value=acf_dist_max)
        evaluation.record_metric(evaluation=name, key='acf_dist_avg', value=acf_dist_avg)
    if show_pacf_distances:
        # display(Markdown("## Show Partial Auto-correlation Distances")
        print("## Show Partial Auto-correlation Distances")
        testing.show_partial_autocorrelation_distances()
    if show_transition_distances:
        # display(Markdown("## Show Transition Distances")
        print("## Show Transition Distances")
        trans_dist_max, trans_dist_avg = testing.show_transition_distances()
        evaluation.record_metric(evaluation=name, key='trans_dist_max', value=trans_dist_max)
        evaluation.record_metric(evaluation=name, key='trans_dist_avg', value=trans_dist_avg)
    if show_series:
        # display(Markdown("## Show Series Sample")
        print("## Show Series Sample")
        testing.show_series(share_ids=True)
    if show_acf:
        # display(Markdown("## Show Series ACF")
        print("## Show Series ACF")
        testing.show_auto_associations()
    return testing


def baseline_evaluation_and_plot(data, evaluation_name, evaluation, ax=None):

    # display(Markdown("data length {}".format(len(data)))
    print("data length {}".format(len(data)))
    train, test = train_test_split(data, test_size=0.5)

    synthesizer = HighDimSynthesizer(df=data)
    testing = UtilityTesting(synthesizer, test, train, test)

    results = testing.show_standard_metrics(ax=ax)

    print_line = ''
    for k, v in results.items():
        print_line += '\n\t{}={:.4f}'.format(k, v)
    print(print_line)

    try:
        utility = testing.utility(target=evaluation.configs[evaluation_name]['target'])
    except Exception as e:
        utility = 1.0
        print(e)

    print(f'Utility: {utility:.3f}')

    if evaluation:
        assert evaluation_name, 'If evaluation is given, evaluation_name must be given too.'
        for k, v in results.items():
            evaluation.record_metric(evaluation=evaluation_name, key=k, value=v)
        evaluation.record_metric(evaluation=evaluation_name, key='utility', value=utility)
        evaluation.record_metric(evaluation=evaluation_name, key='training_time', value=0.)
