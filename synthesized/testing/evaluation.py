import datetime
import time
from collections import OrderedDict
from typing import List, Optional

try:
    from IPython.display import Markdown, display
except ImportError:
    Markdown = str
    display = print
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import simplejson
from matplotlib.axes import Axes
from sklearn.model_selection import train_test_split

from .assessor import Assessor
from ..complex.highdim import HighDimConfig, HighDimSynthesizer
from ..insight import metrics
from ..metadata.factory import MetaExtractor

MAX_PVAL = 0.05


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
        data: pd.DataFrame, name: str, evaluation, config, eval_metrics: List[metrics.TwoDataFrameMetric] = None,
        test_data: Optional[pd.DataFrame] = None, plot_basic: bool = True, plot_losses: bool = False,
        plot_distances: bool = False, show_distributions: bool = False, show_distribution_distances: bool = False,
        show_emd_distances: bool = False, show_correlation_distances: bool = False,
        show_correlation_matrix: bool = False, show_cramers_v_distances: bool = False,
        show_cramers_v_matrix: bool = False, show_logistic_rsquared_distances: bool = False,
        show_logistic_rsquared_matrix: bool = False
) -> Assessor:
    """
    Synthesize and plot data from a Synthesizer trained on the dataframe `data`.
    """
    eval_data = test_data if test_data is not None else data
    len_eval_data = len(eval_data)
    eval_metrics = eval_metrics or []

    def callback(synth, iteration, losses):
        if len(losses) > 0 and hasattr(list(losses.values())[0], 'numpy'):
            if len(synth._loss_history) == 0:
                synth._loss_history.append({n: l.numpy() for n, l in losses.items()})
            else:
                synth._loss_history.append({local_name: losses[local_name].numpy()
                                           for local_name in synth._loss_history[0]})
        return False

    evaluation.record_config(evaluation=name, config=config)
    start = time.time()

    df_meta = MetaExtractor.extract(df=data)
    hd_config = HighDimConfig(**config['params'])
    synthesizer = HighDimSynthesizer(df_meta=df_meta, config=hd_config)
    synthesizer.learn(df_train=data, num_iterations=config['num_iterations'], callback=callback,
                      callback_freq=100)
    training_time = time.time() - start
    synthesized = synthesizer.synthesize(num_rows=len_eval_data)
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
        else:
            display(Markdown("## Plot data"))
            fig, ax = plt.subplots(figsize=(15, 5))
            plot_multidimensional(original=data, synthetic=synthesized, ax=ax)

    assessor = Assessor(df_meta)

    if plot_losses:
        display(Markdown("## Show loss history"))
        df_losses = pd.DataFrame.from_records(synthesizer._loss_history)
        if len(df_losses) > 0:
            df_losses.plot(figsize=(15, 7))

    if plot_distances:
        display(Markdown("## Show average distances"))
        results = assessor.show_standard_metrics(eval_data, synthesized)

        print_line = ''
        for k, v in results.items():
            if evaluation:
                assert name, 'If evaluation is given, evaluation_name must be given too.'
                evaluation.record_metric(evaluation=name, key=k, value=v)
            print_line += '\n\t{}={:.4f}'.format(k, v)
        print(print_line)

    if show_distributions:
        display(Markdown("## Show distributions"))
        assessor.show_distributions(eval_data, synthesized, remove_outliers=0.01)

    # First order metrics
    if show_distribution_distances:
        display(Markdown("## Show distribution distances"))
        assessor.show_ks_distances(eval_data, synthesized)
    if show_emd_distances:
        display(Markdown("## Show EMD distances"))
        assessor.show_emd_distances(eval_data, synthesized)

    # Second order metrics
    if show_correlation_distances:
        display(Markdown("## Show correlation distances"))
        assessor.show_kendall_tau_distances(eval_data, synthesized, max_p_value=MAX_PVAL)
    if show_correlation_matrix:
        display(Markdown("## Show correlation matrices"))
        assessor.show_kendall_tau_matrices(eval_data, synthesized, max_p_value=MAX_PVAL)
    if show_cramers_v_distances:
        display(Markdown("## Show Cramer's V distances"))
        assessor.show_cramers_v_distances(eval_data, synthesized)
    if show_cramers_v_matrix:
        display(Markdown("## Show Cramer's V matrices"))
        assessor.show_cramers_v_matrices(eval_data, synthesized)
    if show_logistic_rsquared_distances:
        display(Markdown("## Show Logistic R^2 distances"))
        assessor.show_categorical_logistic_r2_distances(eval_data, synthesized)
    if show_logistic_rsquared_matrix:
        display(Markdown("## Show Logistic R^2 matrices"))
        assessor.show_categorical_logistic_r2_matrices(eval_data, synthesized)

    return assessor


def baseline_evaluation_and_plot(data, evaluation_name, evaluation, ax=None):

    # display(Markdown("data length {}".format(len(data)))
    print("data length {}".format(len(data)))
    train, test = train_test_split(data, test_size=0.5)
    df_meta = MetaExtractor.extract(data)
    assessor = Assessor(df_meta)

    results = assessor.show_standard_metrics(train, test, ax=ax)

    print_line = ''
    for k, v in results.items():
        print_line += '\n\t{}={:.4f}'.format(k, v)
    print(print_line)

    try:
        utility = assessor.utility(train, test, target=evaluation.configs[evaluation_name]['target'])
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
            crosstab = pd.crosstab(data['x'], columns=[data['y']]).apply(lambda r: r / r.sum(), axis=1)
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
    kolmogorov_smirnov_distance = metrics.KolmogorovSmirnovDistance()
    distances = [kolmogorov_smirnov_distance(original[col], synthetic[col]) for col in original.columns]
    plot = sns.barplot(x=columns, y=distances, hue=dtypes, ax=ax, palette=color_dict, dodge=False)
    plot.set_xticklabels(plot.get_xticklabels(), rotation=30)
    plot.set_title("KS distance by column")
    return plot
