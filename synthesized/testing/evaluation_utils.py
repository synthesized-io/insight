import time
import synthesized
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import ks_2samp
from matplotlib.axes import Axes
from scipy.stats import spearmanr
from synthesized import HighDimSynthesizer
from synthesized.testing import UtilityTesting
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.formula.api import mnlogit, ols
from synthesized.testing.evaluation import Evaluation


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
            return sns.distplot(data, ax=ax)
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


def plot_auto_association(original: np.array, synthetic: np.array, axes: np.array = None):
    lags = list(range(original.shape[-1]))
    axes[0].stem(lags, original, "g", markerfmt='go', use_line_collection=True)
    axes[0].set_title("Original")
    axes[1].stem(lags, synthetic, "b", markerfmt='bo', use_line_collection=True)
    axes[1].set_title("Synthetic")


def plot_avg_distances(test: pd.DataFrame, evaluation: Evaluation, evaluation_name: str,
                       results: list):
    result = []
    for i, (synthesizer, synthesized_, _) in enumerate(results):
        test_ = synthesizer.preprocess(test)
        synthesized_ = synthesizer.preprocess(synthesized_)
        distances = [ks_2samp(test_[col], synthesized_[col])[0] for col in synthesized_.columns]
        avg_distance = np.mean(distances)
        print('run: {}, AVG distance: {}'.format(i+1, avg_distance))
        result.append({'run': i+1, 'avg_distance': avg_distance})
        evaluation.record_metric(evaluation=evaluation_name, key='avg_distance', value=avg_distance)
    df = pd.DataFrame.from_records(result)
    df['run'] = df['run'].astype('category')
    g = sns.barplot(y='run', x='avg_distance', data=df)
    g.set_xlim(0.0, 1.0)


def sequence_index_plot(x, t, ax: Axes, cmap_name: str = "YlGn"):
    values = np.unique(x)
    val2idx = {val: i for i, val in enumerate(values)}
    cmap = cm.get_cmap(cmap_name)
    colors = [cmap(j/values.shape[0]) for j in range(values.shape[0])]

    for i, val in enumerate(x):
        ax.fill_between((i, i+1), 2, facecolor=colors[val2idx[val]])
    #ax.set_xticks(np.arange(0.5, len(t)+0.5, 1), t)
    ax.get_yaxis().set_visible(False)


def sequence_line_plot(x, t, ax):
    sns.lineplot(x=t, y=x, ax=ax)


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
    cat_distances = [np.abs(calculate_auto_association(orig, col, max_order) - calculate_auto_association(synth, col, max_order)).max()
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


# -- training functions
def synthesize_and_plot(data: pd.DataFrame, name: str, evaluation, config, metrics: dict,
                        time_series: bool = False, col: str = "x", max_lag: int = 10, show_anova: bool = False,
                        show_cat_rsquared: bool = False):
    """
    Synthesize and plot data from a `HighDimSynthesizer` trained on the dataframe `data`.
    """
    evaluation.record_config(evaluation=name, config=config)
    start = time.time()
    with HighDimSynthesizer(df=data, **config['params']) as synthesizer:
        synthesizer.learn(df_train=data, num_iterations=config['num_iterations'])
        synthesized = synthesizer.synthesize(num_rows=len(data))
        print('took', time.time() - start, 's')
        print("Metrics:")
        for key, metric in metrics.items():
            value = metric(orig=data, synth=synthesized)
            evaluation.record_metric(evaluation=name, key=key, value=value)
            print(f"{key}: {value}")

        if time_series:
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
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), sharex=True, sharey=True)
            ax1.set_title('orig')
            ax2.set_title('synth')
            plot_data(data, ax=ax1)
            plot_data(synthesized, ax=ax2)
        else:
            fig, ax = plt.subplots(figsize=(15, 5))
            plot_multidimensional(original=data, synthetic=synthesized, ax=ax)
        if show_anova:
            testing = UtilityTesting(synthesizer, data, data, synthesized)
            testing.show_anova()
        if show_cat_rsquared:
            testing = UtilityTesting(synthesizer, data, data, synthesized)
            testing.show_categorical_rsquared()


def train_and_synthesize(data: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame,
                         evaluation: Evaluation, evaluation_name: str):
    synthesizer_class = evaluation.configs[evaluation_name]["synthesizer_class"]
    assert synthesizer_class in {"HighDimSynthesizer", "SeriesSynthesizer"}
    synthesizer_constructor = getattr(synthesized, synthesizer_class)
    loss_history = list()

    def callback(synth, iteration, losses):
        if len(loss_history) == 0:
            loss_history.append(losses)
        else:
            loss_history.append({name: losses[name] for name in loss_history[0]})

    with synthesizer_constructor(df=data, **evaluation.configs[evaluation_name]['params']) as synthesizer:
        synthesizer.learn(
            df_train=train, num_iterations=evaluation.configs[evaluation_name]['num_iterations'],
            callback=callback, callback_freq=100
        )
        if synthesizer_class == "HighDimSynthesizer":
            synthesized_data = synthesizer.synthesize(num_rows=len(test))
        else:
            series_lengths = data.groupby(
                evaluation.configs[evaluation_name]["params"]["identifier_label"]).count().to_numpy().transpose()
            series_lengths = list(series_lengths[0])
            synthesized_data = synthesizer.synthesize(series_lengths=series_lengths)
        return synthesizer, synthesized_data, loss_history



