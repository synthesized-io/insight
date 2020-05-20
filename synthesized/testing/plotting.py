import logging
import time
from typing import Optional, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display
from matplotlib import cm
from matplotlib.axes import Axes
from scipy.stats import ks_2samp, spearmanr
from statsmodels.formula.api import mnlogit, ols
from statsmodels.tsa.stattools import acf, pacf

from .metrics import calculate_evaluation_metrics
from ..highdim import HighDimSynthesizer
from ..series import SeriesSynthesizer
from ..testing import UtilityTesting
from ..testing.evaluation import Evaluation
from ..insight import metrics

logger = logging.getLogger(__name__)

MAX_SAMPLE_DATES = 2500
NUM_UNIQUE_CATEGORICAL = 100
MAX_PVAL = 0.05
NAN_FRACTION_THRESHOLD = 0.25
NON_NAN_COUNT_THRESHOLD = 500
CATEGORICAL_THRESHOLD_LOG_MULTIPLIER = 2.5


# -- Plotting functions
def plot_data(data: pd.DataFrame, ax: Axes):
    """Plot one- or two-dimensional dataframe `data` on `matplotlib` axis `ax` according to column types. """
    if data.shape[1] == 1:
        if data['x'].dtype.kind == 'O':
            return sns.countplot(data["x"], ax=ax)
        else:
            return sns.distplot(data["x"], ax=ax)
    elif data.shape[1] == 2:
        if data['x'].dtype.kind in {'O', 'i'} and data['y'].dtype.kind == 'f':
            sns.violinplot(x="x", y="y", data=data, ax=ax)
        elif data['x'].dtype.kind == 'f' and data['y'].dtype.kind == 'f':
            return ax.hist2d(data['x'], data['y'], bins=100)
        elif data['x'].dtype.kind == 'O' and data['y'].dtype.kind == 'O':
            crosstab = pd.crosstab(data['x'], columns=[data['y']]).apply(lambda r: r/r.sum(), axis=1)
            sns.heatmap(crosstab, vmin=0.0, vmax=1.0, ax=ax)
        else:
            return sns.distplot(data, ax=ax, color=["b", "g"])
    else:
        return sns.distplot(data, ax=ax)


def plot_multidimensional(original: pd.DataFrame, synthetic: pd.DataFrame, ax: Axes = None):
    """
    Plot Kolmogorov-Smirnov distance between the columns in the dataframes
    `original` and `synthetic` on `matplotlib` axis `ax`.
    """
    dtype_dict = {"O": "Categorical", "i": "Categorical", "f": "Continuous"}
    default_palette = sns.color_palette()
    color_dict = {"Categorical": default_palette[0], "Continuous": default_palette[1]}
    assert (original.columns == synthetic.columns).all(), "Original and synthetic data must have the same columns."
    columns = original.columns.values.tolist()
    error_msg = "Original and synthetic data must have the same data types."
    assert (original.dtypes.values == synthetic.dtypes.values).all(), error_msg
    dtypes = [dtype_dict[dtype.kind] for dtype in original.dtypes.values]
    distances = [ks_2samp(original[col], synthetic[col])[0] for col in original.columns]
    plot = sns.barplot(x=columns, y=distances, hue=dtypes, ax=ax, palette=color_dict, dodge=False)
    plot.set_xticklabels(plot.get_xticklabels(), rotation=30)
    plot.set_title("KS distance by column")
    return plot


def plot_time_series(x, t, ax):
    kind = x.dtype.kind
    if kind in {"i", "f"}:
        sequence_line_plot(x=x, t=t, ax=ax)
    else:
        sequence_index_plot(x=x, t=t, ax=ax)


def plot_auto_association(original: np.array, synthetic: np.array, axes: np.array):
    assert axes is not None
    lags = list(range(original.shape[-1]))
    axes[0].stem(lags, original, "g", markerfmt='go', use_line_collection=True)
    axes[0].set_title("Original")
    axes[1].stem(lags, synthetic, "b", markerfmt='bo', use_line_collection=True)
    axes[1].set_title("Synthetic")


def calculate_mean_max(x: List[float]) -> Tuple[float, float]:
    if len(x) > 0:
        return np.nanmean(x), np.nanmax(x)
    else:
        return 0., 0.


def plot_avg_distances(synthesized: pd.DataFrame, test: pd.DataFrame,
                       evaluation: Optional[Evaluation] = None, evaluation_name: Optional[str] = None,
                       ax: Axes = None):
    test = test.copy()
    synthesized = synthesized.copy()

    metrics = calculate_evaluation_metrics(df_orig=test, df_synth=synthesized)
    print(metrics)

    # Compute summaries
    avg_ks_distance, max_ks_distance = calculate_mean_max(metrics['ks_distances'])
    avg_corr_distance, max_corr_distance = calculate_mean_max(metrics['corr_distances'].values)
    avg_emd_distance, max_emd_distance = calculate_mean_max(metrics['emd_distances'])
    avg_cramers_v_distance, max_cramers_v_distance = calculate_mean_max(metrics['cramers_v_distances'].values)
    avg_log_corr_distance, max_log_corr_distance = calculate_mean_max(metrics['logistic_corr_distances'].values)

    current_result = {
        'ks_distance_avg': avg_ks_distance,
        'ks_distance_max': max_ks_distance,
        'emd_categ_avg': avg_emd_distance,
        'emd_categ_max': max_emd_distance,
        'corr_dist_avg': avg_corr_distance,
        'corr_dist_max': max_corr_distance,
        'cramers_v_avg': avg_cramers_v_distance,
        'cramers_v_max': max_cramers_v_distance,
        'logistic_corr_dist_avg': avg_log_corr_distance,
        'logistic_corr_dist_max': max_log_corr_distance
    }
    print(current_result)

    print_line = ''
    for k, v in current_result.items():
        if evaluation:
            assert evaluation_name, 'If evaluation is given, evaluation_name must be given too.'
            evaluation.record_metric(evaluation=evaluation_name, key=k, value=v)
        print_line += '\n\t{}={:.4f}'.format(k, v)

    g = sns.barplot(x=list(current_result.keys()), y=list(current_result.values()), ax=ax, palette='Paired')

    values = list(current_result.values())
    for i in range(len(values)):
        v = values[i]
        g.text(i, v, round(v, 3), color='black', ha="center")

    if ax:
        for tick in ax.get_xticklabels():
            tick.set_rotation(10)
    else:
        plt.xticks(rotation=10)


def sequence_index_plot(x, t, ax: Axes, cmap_name: str = "YlGn"):
    values = np.unique(x)
    val2idx = {val: i for i, val in enumerate(values)}
    cmap = cm.get_cmap(cmap_name)
    colors = [cmap(j/values.shape[0]) for j in range(values.shape[0])]

    for i, val in enumerate(x):
        ax.fill_between((i, i+1), 2, facecolor=colors[val2idx[val]])
    ax.get_yaxis().set_visible(False)


def sequence_line_plot(x, t, ax):
    sns.lineplot(x=t, y=x, ax=ax)


# -- training functions
def synthesize_and_plot(data: pd.DataFrame, name: str, evaluation, config, eval_metrics: dict,
                        test_data: Optional[pd.DataFrame] = None, time_series: bool = False,
                        col: str = "x", max_lag: int = 10, plot_basic: bool = True, plot_losses: bool = False,
                        plot_distances: bool = False, show_distributions: bool = False,
                        show_distribution_distances: bool = False, show_emd_distances: bool = False,
                        show_correlation_distances: bool = False, show_correlation_matrix: bool = False,
                        show_cramers_v_distances: bool = False, show_cramers_v_matrix: bool = False,
                        show_cat_rsquared: bool = False,
                        show_acf_distances: bool = False,  show_pacf_distances: bool = False,
                        show_transition_distances: bool = False, show_series: bool = False):
    """
    Synthesize and plot data from a Synthesizer trained on the dataframe `data`.
    """
    eval_data = test_data if test_data is not None else data
    len_eval_data = len(eval_data)

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

    if config['synthesizer_class'] == 'HighDimSynthesizer':

        with HighDimSynthesizer(df=data, **config['params']) as synthesizer:
            synthesizer.learn(df_train=data, num_iterations=config['num_iterations'], callback=callback,
                              callback_freq=100)
            training_time = time.time() - start
            synthesized = synthesizer.synthesize(num_rows=len_eval_data)
            value_factory = synthesizer.value_factory
            print('took', training_time, 's')

    elif config['synthesizer_class'] == 'SeriesSynthesizer':

        if 'identifier_label' in config['params'].keys() and config['params']['identifier_label'] is not None:
            identifier_label = config['params']['identifier_label']
            num_series = eval_data[identifier_label].nunique()
            series_length = int(len_eval_data / num_series)
        else:
            num_series = 1
            series_length = len_eval_data

        with SeriesSynthesizer(df=data, **config['params']) as synthesizer:
            synthesizer.learn(df_train=data, num_iterations=config['num_iterations'], callback=callback,
                              callback_freq=100)
            training_time = time.time() - start
            synthesized = synthesizer.synthesize(num_series=num_series, series_length=series_length)
            value_factory = synthesizer.value_factory
            print('took', training_time, 's')

    else:
        raise NotImplementedError("Given 'synthesizer_class={}' not supported.".format(config['synthesizer_class']))

    evaluation.record_metric(evaluation=name, key='training_time', value=training_time)
    print("Metrics:")
    for key, metric in eval_metrics.items():
        value = metric(orig=data, synth=synthesized)
        evaluation.record_metric(evaluation=name, key=key, value=value)
        print(f"{key}: {value}")

    if plot_basic:
        if time_series:
            display(Markdown("## Plot time-series data"))
            fig, axes = plt.subplots(2, 2, figsize=(15, 5), sharey="row")
            fig.tight_layout()
            original_auto_assoc = calculate_auto_association(dataset=data, col=col, max_order=max_lag)
            synthetic_auto_assoc = calculate_auto_association(dataset=synthesized, col=col, max_order=max_lag)
            t_orig = np.arange(0, data.shape[0])
            plot_time_series(x=data[col].to_numpy(), t=t_orig, ax=axes[0, 0])
            axes[0, 0].set_title("Original")
            t_synth = np.arange(0, synthesized.shape[0])
            plot_time_series(x=synthesized[col].to_numpy(), t=t_synth, ax=axes[0, 1])
            axes[0, 1].set_title("Synthetic")
            plot_auto_association(original=original_auto_assoc, synthetic=synthetic_auto_assoc, axes=axes[1])
        elif data.shape[1] <= 3:
            display(Markdown("## Plot data"))
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), sharex=True, sharey=True)
            ax1.set_title('orig')
            ax2.set_title('synth')
            plot_data(data, ax=ax1)
            plot_data(synthesized, ax=ax2)
        else:
            display(Markdown("## Plot data"))
            fig, ax = plt.subplots(figsize=(15, 5))
            plot_multidimensional(original=data, synthetic=synthesized, ax=ax)

    testing = UtilityTesting(synthesizer, data, eval_data, synthesized, identifier=identifier_label)

    if plot_losses:
        display(Markdown("## Show loss history"))
        df_losses = pd.DataFrame.from_records(synthesizer.loss_history)
        if len(df_losses) > 0:
            df_losses.plot(figsize=(15, 7))
        plt.show()

    if plot_distances:
        display(Markdown("## Show average distances"))

        # Calculate 1st order metrics for categorical/continuous
        avg_ks_distance, max_ks_distance = testing.metric_mean_max(metrics.kolmogorov_smirnov_distance)
        avg_emd_distance, max_emd_distance = testing.metric_mean_max(metrics.earth_movers_distance)

        # Calculate 2nd order metrics for categorical/continuous
        avg_corr_distance, max_corr_distance = testing.metric_mean_max(metrics.kendell_tau_correlation, max_p_value=MAX_PVAL)
        avg_cramers_v_distance, max_cramers_v_distance = testing.metric_mean_max(metrics.cramers_v)
        avg_log_corr_distance, max_log_corr_distance = testing.metric_mean_max(metrics.categorical_logistic_correlation, continuous_input_only=True)

        current_result = {
            'ks_distance_avg': avg_ks_distance,
            'ks_distance_max': max_ks_distance,
            'emd_categ_avg': avg_emd_distance,
            'emd_categ_max': max_emd_distance,
            'corr_dist_avg': avg_corr_distance,
            'corr_dist_max': max_corr_distance,
            'cramers_v_avg': avg_cramers_v_distance,
            'cramers_v_max': max_cramers_v_distance,
            'logistic_corr_dist_avg': avg_log_corr_distance,
            'logistic_corr_dist_max': max_log_corr_distance
        }

        print_line = ''
        for k, v in current_result.items():
            if evaluation:
                assert name, 'If evaluation is given, evaluation_name must be given too.'
                evaluation.record_metric(evaluation=name, key=k, value=v)
            print_line += '\n\t{}={:.4f}'.format(k, v)

        testing.show_results(current_result)

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
        display(Markdown("## Show Cramer's V matrices"), Markdown("## Show Cramer's V matrices"))
        testing.show_second_order_metric_matrices(metrics.cramers_v)
    if show_cat_rsquared:
        display(Markdown("## Show categorical R^2"))
        testing.show_second_order_metric_matrices(metrics.categorical_logistic_correlation, continuous_inputs_only=True)

    # TIME SERIES
    if show_acf_distances:
        display(Markdown("## Show Auto-correaltion Distances"))
        acf_dist_max, acf_dist_avg = testing.show_autocorrelation_distances()
        evaluation.record_metric(evaluation=name, key='acf_dist_max', value=acf_dist_max)
        evaluation.record_metric(evaluation=name, key='acf_dist_avg', value=acf_dist_avg)
        plt.show()
    if show_pacf_distances:
        display(Markdown("## Show Partial Auto-correlation Distances"))
        testing.show_partial_autocorrelation_distances()
        plt.show()
    if show_transition_distances:
        display(Markdown("## Show Transition Distances"))
        trans_dist_max, trans_dist_avg = testing.show_transition_distances()
        evaluation.record_metric(evaluation=name, key='trans_dist_max', value=trans_dist_max)
        evaluation.record_metric(evaluation=name, key='trans_dist_avg', value=trans_dist_avg)
        plt.show()
    if show_series:
        display(Markdown("## Show Series Sample"))
        testing.show_series()
        plt.show()
    return testing


# -- Measures of association for different pairs of data types
def calculate_auto_association(dataset: pd.DataFrame, col: str, max_order: int):
    variable = dataset[col].to_numpy()
    association = association_dict[variable.dtype.kind]
    auto_associations = []
    for order in range(1, max_order+1):
        postfix = variable[order:]
        prefix = variable[:-order]
        auto_associations.append(association(x=prefix, y=postfix))
    return np.array(auto_associations)


# --- Association between continuous and continuous
def continuous_correlation(x, y):
    return np.corrcoef(x, y)[0, 1]


def ordered_correlation(x, y):
    return spearmanr(x, y).correlation


# --- Association between categorical and categorical
def categorical_logistic_rsquared(x, y):
    temp_df = pd.DataFrame({"x": x, "y": y})
    model = mnlogit("y ~ C(x)", data=temp_df).fit(method="cg", disp=0)
    return model.prsquared


def max_autocorrelation_distance(orig: pd.DataFrame, synth: pd.DataFrame):
    floats = [col for dtype, col in zip(orig.dtypes.values, orig.columns.values)
              if dtype.kind == "f"]
    acf_distances = [np.abs((acf(orig[col], fft=True) - acf(synth[col], fft=True))).max()
                     for col in floats]
    return max(acf_distances)


def max_partial_autocorrelation_distance(orig: pd.DataFrame, synth: pd.DataFrame):
    floats = [col for dtype, col in zip(orig.dtypes.values, orig.columns.values)
              if dtype.kind == "f"]
    pacf_distances = [np.abs((pacf(orig[col]) - pacf(synth[col]))).max()
                      for col in floats]
    return max(pacf_distances)


def max_categorical_auto_association_distance(orig: pd.DataFrame, synth: pd.DataFrame, max_order=20):
    cats = [col for dtype, col in zip(orig.dtypes.values, orig.columns.values)
            if dtype.kind == "O"]
    cat_distances = [np.abs(calculate_auto_association(orig, col, max_order) -
                            calculate_auto_association(synth, col, max_order)).max()
                     for col in cats]
    return max(cat_distances)


def mean_squared_error_closure(col, baseline: float = 1):
    def mean_squared_error(orig: pd.DataFrame, synth: pd.DataFrame):
        return ((orig[col].to_numpy() - synth[col].to_numpy())**2).mean()/baseline
    return mean_squared_error


def rolling_mse_asof(sd, time_unit=None):
    """
    Calculate the mean-squared error between the "x" values of the original and synthetic
    data. The sets of times may not be identical so we use "as of" (last observation rolled
    forward) to interpolate between the times in the two datasets.

    The dates are also optionally truncated to some unit following the syntax for the pandas
    `.floor` function.

    :param sd: [float] error standard deviation
    :param time_unit: [str] the time unit to round to. See documentation for pandas `.floor` method.
    :return: [(float, float)] MSE and MSE/(2*error variance)
    """
    # truncate date
    def mse_function(orig, synth):
        if time_unit is not None:
            synth.t = synth.t.dt.floor(time_unit)
            orig.t = orig.t.dt.floor(time_unit)

        # join datasets
        joined = pd.merge_asof(orig[["t", "x"]], synth[["t", "x"]], on="t")

        # calculate metrics
        mse = ((joined.x_x - joined.x_y) ** 2).mean()
        mse_eff = mse / (2 * sd ** 2)

        return mse_eff
    return mse_function


# -- global constants
association_dict = {"i": ordered_correlation, "f": continuous_correlation,
                    "O": categorical_logistic_rsquared}
