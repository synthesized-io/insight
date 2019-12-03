import time
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from typing import Optional

from scipy.stats import ks_2samp
from matplotlib.axes import Axes

import matplotlib.pyplot as plt
from synthesized import HighDimSynthesizer
from ..testing import UtilityTesting
from .evaluation_utils import calculate_auto_association
from synthesized.testing.evaluation import Evaluation
from synthesized.testing.util import categorical_emd

from IPython.display import Markdown


# -- Plotting functions
def plot_data(data: pd.DataFrame, ax: Axes):
    """Plot one- or two-dimensional dataframe `data` on `matplotlib` axis `ax` according to column types. """
    if data.shape[1] == 1:
        if data['x'].dtype.kind == 'O':
            return sns.countplot(data["x"], ax=ax)
        else:
            return sns.distplot(data["x"], ax=ax)
    elif data.shape[1] == 2:
        if data['x'].dtype.kind == 'O' and data['y'].dtype.kind == 'f':
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


def plot_avg_distances(synthesized: pd.DataFrame, test: pd.DataFrame,
                       evaluation: Evaluation, evaluation_name: str):
    synthesized = synthesized.copy().dropna()

    # Calculate distances for all columns
    ks_distances = []
    emd = []
    for col in test.columns:
        if test[col].dtype.kind in ('f', 'i'):
            ks_distances.append(ks_2samp(test[col], synthesized[col])[0])
        else:
            try:
                ks_distances.append(ks_2samp(
                    pd.to_numeric(pd.to_datetime(test[col])),
                    pd.to_numeric(pd.to_datetime(synthesized[col]))
                )[0])
            except ValueError:
                emd.append(categorical_emd(test[col], synthesized[col]))

    corr = np.abs((test.corr() - synthesized.corr()).to_numpy())

    # Compute summaries
    avg_ks_distance = np.mean(ks_distances)
    max_ks_distance = np.max(ks_distances)
    avg_corr = corr.mean()
    max_corr = corr.max()
    if len(emd) > 0:
        avg_emd = np.mean(emd)
        max_emd = np.max(emd)
    else:
        avg_emd = float("nan")
        max_emd = float("nan")

    current_result = {
        'ks_distance_avg': avg_ks_distance,
        'ks_distance_max': max_ks_distance,
        'corr_dist_avg': avg_corr,
        'corr_dist_max': max_corr,
        'emd_categ_avg': avg_emd,
        'emd_categ_max': max_emd
    }

    print_line = ''
    for k, v in current_result.items():
        evaluation.record_metric(evaluation=evaluation_name, key=k, value=v)
        print_line += '\n\t{}={:.4f}'.format(k, v)

    plt.figure()
    sns.barplot(x=list(current_result.keys()), y=list(current_result.values()))
    plt.xticks(rotation=10)
    plt.show()


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
def synthesize_and_plot(data: pd.DataFrame, name: str, evaluation, config, metrics: dict,
                        test_data: Optional[pd.DataFrame] = None, time_series: bool = False,
                        col: str = "x", max_lag: int = 10, plot_losses: bool = False,
                        plot_distances: bool = False, show_distribution_distances: bool = False,
                        show_distributions: bool = False, show_correlation_distances: bool = False,
                        show_correlation_matrix: bool = False, show_anova: bool = False,
                        show_cat_rsquared: bool = False):
    """
    Synthesize and plot data from a `HighDimSynthesizer` trained on the dataframe `data`.
    """
    eval_data = test_data if test_data is not None else data
    loss_history = list()

    def callback(synth, iteration, losses):
        if len(loss_history) == 0:
            loss_history.append(losses)
        else:
            loss_history.append({name: losses[local_name] for local_name in loss_history[0]})
        return True

    evaluation.record_config(evaluation=name, config=config)
    start = time.time()
    with HighDimSynthesizer(df=data, **config['params']) as synthesizer:
        synthesizer.learn(df_train=data, num_iterations=config['num_iterations'], callback=callback, callback_freq=100)
        training_time = time.time() - start
        synthesized = synthesizer.synthesize(num_rows=len(data))
        print('took', training_time, 's')
        evaluation.record_metric(evaluation=name, key='training_time', value=training_time)
        print("Metrics:")
        for key, metric in metrics.items():
            value = metric(orig=data, synth=synthesized)
            evaluation.record_metric(evaluation=name, key=key, value=value)
            print(f"{key}: {value}")

        if time_series:
            Markdown("## Plot time-series data")
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
            Markdown("## Plot data")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), sharex=True, sharey=True)
            ax1.set_title('orig')
            ax2.set_title('synth')
            plot_data(data, ax=ax1)
            plot_data(synthesized, ax=ax2)
        else:
            Markdown("## Plot data")
            fig, ax = plt.subplots(figsize=(15, 5))
            plot_multidimensional(original=data, synthetic=synthesized, ax=ax)
        testing = UtilityTesting(synthesizer, data, eval_data, synthesized)
        if plot_losses:
            Markdown("## Show loss history")
            pd.DataFrame.from_records(loss_history).plot(figsize=(15, 7))
        if plot_distances:
            Markdown("## Show average distances")

            plot_avg_distances(test=eval_data, synthesized=synthesized, evaluation=evaluation, evaluation_name=name)
        if show_distribution_distances:
            Markdown("## Show distribution distances")
            testing.show_distribution_distances()
        if show_distributions:
            Markdown("## Show distributions")
            testing.show_distributions(remove_outliers=0.01)
        if show_correlation_distances:
            Markdown("## Show correlation distances")
            testing.show_corr_distances()
        if show_correlation_matrix:
            Markdown("## Show correlation matrices")
            testing.show_corr_matrices()
        if show_anova:
            Markdown("## Show correlation matrices")
            testing.show_anova()
        if show_cat_rsquared:
            testing.show_categorical_rsquared()
        return testing
