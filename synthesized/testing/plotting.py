import time
from typing import Optional, List, Tuple
import logging

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, spearmanr, kendalltau
from statsmodels.formula.api import mnlogit, ols
from statsmodels.tsa.stattools import acf, pacf
from IPython.display import Markdown, display

from ..highdim import HighDimSynthesizer
from ..testing import UtilityTesting
from ..testing.evaluation import Evaluation
from ..testing.util import categorical_emd


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
        return np.mean(x), np.max(x)
    else:
        return 0., 0.


def plot_avg_distances(synthesized: pd.DataFrame, test: pd.DataFrame,
                       evaluation: Evaluation, evaluation_name: str, ax: Axes = None,
                       record_metrics: bool = True):
    test = test.copy()
    synthesized = synthesized.copy()

    # Calculate distances for all columns
    ks_distances = []
    emd_distances = []
    corr_distances = []
    numerical_columns = []

    for col in test.columns:
        ks_distance, emd_distance = None, None
        len_test = len(test)

        if test[col].dtype.kind == 'f':
            col_test_clean = test[col].dropna()
            col_synth_clean = synthesized[col].dropna()
            if len(col_test_clean) < len_test:
                logger.info("Column '{}' contains NaNs. Computing KS distance with {}/{} samples"
                            .format(col, len(col_test_clean), len_test))
            ks_distance, _ = ks_2samp(col_test_clean, col_synth_clean)
            ks_distances.append(ks_distance)
            numerical_columns.append(col)

        elif test[col].dtype.kind == 'i':
            if test[col].nunique() < np.log(len(test)) * CATEGORICAL_THRESHOLD_LOG_MULTIPLIER:
                logger.info("Column '{}' treated as categorical with {} categories".format(col, test[col].nunique()))
                emd_distance = categorical_emd(test[col].dropna(), synthesized[col].dropna())
                emd_distances.append(emd_distance)

            else:
                col_test_clean = test[col].dropna()
                col_synth_clean = synthesized[col].dropna()
                if len(col_test_clean) < len_test:
                    logger.info("Column '{}' contains NaNs. Computing KS distance with {}/{} samples"
                                .format(col, len(col_test_clean), len_test))
                ks_distance, _ = ks_2samp(col_test_clean, col_synth_clean)
                ks_distances.append(ks_distance)

            numerical_columns.append(col)

        elif test[col].dtype.kind in ('O', 'b'):

            # Try to convert to numeric
            col_num = pd.to_numeric(test[col], errors='coerce')
            if col_num.isna().sum() / len(col_num) < NAN_FRACTION_THRESHOLD:
                test[col] = col_num
                synthesized[col] = pd.to_numeric(synthesized[col], errors='coerce')
                numerical_columns.append(col)

            # if (is not sampling) and (is not date):
            elif (test[col].nunique() <= np.log(len(test)) * CATEGORICAL_THRESHOLD_LOG_MULTIPLIER) and \
                    np.all(pd.to_datetime(test[col].sample(min(len(test), MAX_SAMPLE_DATES)), errors='coerce').isna()):
                emd_distance = categorical_emd(test[col].dropna(), synthesized[col].dropna())
                if not np.isnan(emd_distance):
                    emd_distances.append(emd_distance)

    for i in range(len(numerical_columns)):
        col_i = numerical_columns[i]
        for j in range(i + 1, len(numerical_columns)):
            col_j = numerical_columns[j]
            test_clean = test[[col_i, col_j]].dropna()
            synth_clean = synthesized[[col_i, col_j]].dropna()

            if len(test_clean) > 0 and len(synth_clean) > 0:
                corr_orig, pvalue_orig = kendalltau(test_clean[col_i].values, test_clean[col_j].values)
                corr_synth, pvalue_synth = kendalltau(synth_clean[col_i].values, synth_clean[col_j].values)

                if pvalue_orig <= MAX_PVAL or pvalue_synth <= MAX_PVAL:
                    corr_distances.append(abs(corr_orig - corr_synth))

    # Compute summaries
    avg_ks_distance, max_ks_distance = calculate_mean_max(ks_distances)
    avg_corr_distance, max_corr_distance = calculate_mean_max(corr_distances)
    avg_emd_distance, max_emd_distance = calculate_mean_max(emd_distances)

    current_result = {
        'ks_distance_avg': avg_ks_distance,
        'ks_distance_max': max_ks_distance,
        'corr_dist_avg': avg_corr_distance,
        'corr_dist_max': max_corr_distance,
        'emd_categ_avg': avg_emd_distance,
        'emd_categ_max': max_emd_distance
    }

    print_line = ''
    for k, v in current_result.items():
        if record_metrics:
            evaluation.record_metric(evaluation=evaluation_name, key=k, value=v)
        print_line += '\n\t{}={:.4f}'.format(k, v)

    g = sns.barplot(x=list(current_result.keys()), y=list(current_result.values()), ax=ax)

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
def synthesize_and_plot(data: pd.DataFrame, name: str, evaluation, config, metrics: dict,
                        test_data: Optional[pd.DataFrame] = None, time_series: bool = False,
                        col: str = "x", max_lag: int = 10, plot_basic: bool = True, plot_losses: bool = False,
                        plot_distances: bool = False, show_distribution_distances: bool = False,
                        show_distributions: bool = False, show_correlation_distances: bool = False,
                        show_correlation_matrix: bool = False, show_anova: bool = False,
                        show_cat_rsquared: bool = False):
    """
    Synthesize and plot data from a `HighDimSynthesizer` trained on the dataframe `data`.
    """
    eval_data = test_data if test_data is not None else data
    loss_history: List[dict] = list()

    def callback(synth, iteration, losses):
        if len(loss_history) == 0:
            loss_history.append(losses)
        else:
            loss_history.append({local_name: losses[local_name] for local_name in loss_history[0]})
        return False

    evaluation.record_config(evaluation=name, config=config)
    start = time.time()
    with HighDimSynthesizer(df=data, **config['params']) as synthesizer:
        synthesizer.learn(df_train=data, num_iterations=config['num_iterations'], callback=callback, callback_freq=100)
        training_time = time.time() - start
        synthesized = synthesizer.synthesize(num_rows=len(eval_data))
        print('took', training_time, 's')
        evaluation.record_metric(evaluation=name, key='training_time', value=training_time)
        print("Metrics:")
        for key, metric in metrics.items():
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
        testing = UtilityTesting(synthesizer, data, eval_data, synthesized)
        if plot_losses:
            display(Markdown("## Show loss history"))
            pd.DataFrame.from_records(loss_history).plot(figsize=(15, 7))
            plt.show()
        if plot_distances:
            display(Markdown("## Show average distances"))
            plot_avg_distances(test=eval_data, synthesized=synthesized, evaluation=evaluation, evaluation_name=name)
            plt.show()
        if show_distribution_distances:
            display(Markdown("## Show distribution distances"))
            testing.show_distribution_distances()
            plt.show()
        if show_distributions:
            display(Markdown("## Show distributions"))
            testing.show_distributions(remove_outliers=0.01)
            plt.show()
        if show_correlation_distances:
            display(Markdown("## Show correlation distances"))
            testing.show_corr_distances()
            plt.show()
        if show_correlation_matrix:
            display(Markdown("## Show correlation matrices"))
            testing.show_corr_matrices()
            plt.show()
        if show_anova:
            display(Markdown("## Show correlation matrices"))
            testing.show_anova()
            plt.show()
        if show_cat_rsquared:
            display(Markdown("## Show categorical R^2"))
            testing.show_categorical_rsquared()
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


def continuous_rsquared(x, y):
    return continuous_correlation(x, y)**2


def ordered_correlation(x, y):
    return spearmanr(x, y).correlation


def ordered_rsquared(x, y):
    return ordered_correlation(x, y)**2


# --- Association between categorical and categorical
def categorical_logistic_rsquared(x, y):
    temp_df = pd.DataFrame({"x": x, "y": y})
    model = mnlogit("y ~ C(x)", data=temp_df).fit(method="cg", disp=0)
    return model.prsquared


# --- Association between continuous and categorical
def continuous_logistic_rsquared(x, y):
    temp_df = pd.DataFrame({"x": x, "y": y})
    model = mnlogit("y ~ x", data=temp_df).fit(method="cg", disp=0)
    return model.prsquared


# --- Association between categorical and continuous
def anova_rsquared(x, y):
    temp_df = pd.DataFrame({"x": x, "y": y})
    model = ols("y ~ C(x)", data=temp_df).fit(method="cg", disp=0)
    return model.rsquared


# -- Evaluation metrics
def max_correlation_distance(orig: pd.DataFrame, synth: pd.DataFrame):
    return np.abs((orig.corr() - synth.corr()).to_numpy()).max()


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


def mean_ks_distance(orig: pd.DataFrame, synth: pd.DataFrame):
    distances = [ks_2samp(orig[col], synth[col])[0] for col in orig.columns]
    return np.mean(distances)


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
default_metrics = {"avg_distance": mean_ks_distance}
association_dict = {"i": ordered_correlation, "f": continuous_correlation,
                    "O": categorical_logistic_rsquared}
