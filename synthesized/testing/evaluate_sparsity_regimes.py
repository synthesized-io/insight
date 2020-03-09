import os
import time

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import simplejson
from scipy.stats import ks_2samp
from sklearn.preprocessing import QuantileTransformer, StandardScaler, LabelEncoder

from ..highdim.highdim import HighDimSynthesizer


def missing_patterns(data: pd.DataFrame, keep_ratio: float, mechanism: str = 'MCAR',
                     std_noise: float = 1.) -> pd.DataFrame:

    data_missing = data.copy()

    if mechanism == 'MCAR':
        def drop_rand(num):
            return np.nan if np.random.uniform() > keep_ratio else num

        for column in data_missing.columns:
            data_missing[column] = data_missing[column].apply(drop_rand)

        return data_missing

    elif mechanism in ('MAR', 'MNAR'):

        for c in data.columns:
            if data[c].dtype.kind not in ('i', 'f'):
                le = LabelEncoder()
                data[c] = le.fit_transform(data[c])
                data_missing[c] = le.fit_transform(data_missing[c])

        for c in data_missing.columns:

            if mechanism == 'MAR':
                X = np.array(data_missing[list(filter(lambda x: x != c, data_missing.columns))])
            elif mechanism == 'MNAR':
                X = np.array(data_missing)

            # Scale to 0 mean, 1 std
            ss = StandardScaler()
            X = ss.fit_transform(X)

            w = np.ones((X.shape[1]))
            data_missing['drop_prob'] = np.dot(w, X.T)

            # Add noise
            data_missing['drop_prob'] = (data_missing['drop_prob'] + np.random.normal(0, std_noise, len(data_missing)))

            # Transform to uniform(0,1)
            qt = QuantileTransformer()
            data_missing['drop_prob'] = qt.fit_transform(data_missing['drop_prob'].values.reshape(-1, 1))

            # Drop values
            data_missing.loc[keep_ratio < data_missing['drop_prob'], c] = np.nan

            del data_missing['drop_prob']

        return data_missing

    else:
        raise NotImplementedError


def plot_missing_patterns(data, plot_function, keep_ratio=0.5, std_noise=2.5, mechanisms=None):
    if not mechanisms:
        mechanisms = ['MCAR', 'MAR', 'MNAR']

    for mechanism in mechanisms:
        print('============= MCAR =============')
        data_missing = missing_patterns(data, keep_ratio=keep_ratio, std_noise=std_noise, mechanism=mechanism)

        print('Prop. NaNs:')
        print(np.sum(data_missing.isna()) / len(data_missing))
        plot_function(data, data_missing, label1='Original', label2='Missing', title=mechanism)


def max_correlation_distance(orig, synth):
    return np.abs((orig.corr() - synth.corr()).to_numpy()).max()


def mean_correlation_distance(orig, synth):
    return np.abs((orig.corr() - synth.corr()).to_numpy()).mean()


def mean_ks_distance(orig, synth):
    distances = [ks_2samp(orig[col], synth[col])[0] for col in orig.columns]
    return np.mean(distances)


def max_ks_distance(orig, synth):
    distances = [ks_2samp(orig[col], synth[col])[0] for col in orig.columns]
    return np.max(distances)


def synthesize_and_plot_results(data: pd.DataFrame, mechanism: str = 'MCAR', n_iter: int = 2500,
                                std_noise: float = 2., start: int = 25, end: int = 100, step: int = 5,
                                n_experiments: int = 3):
    max_ks_vec = []
    mean_ks_vec = []
    max_corr_vec = []
    mean_corr_vec = []
    keep_ratio_vec = []

    for i in range(start, end + 1, step):
        keep_ratio = i / 100.0
        t_start = time.time()

        keep_ratio_vec.append(keep_ratio)
        max_ks_iter = []
        mean_ks_iter = []
        max_corr_iter = []
        mean_corr_iter = []

        for j in range(n_experiments):
            data_missing = missing_patterns(data, keep_ratio=keep_ratio, mechanism=mechanism, std_noise=std_noise)

            config_path = os.environ.get('EVALUATION_CONFIG_PATH', "configs/evaluation/synthetic_distributions.json")
            with open(config_path, 'r') as f:
                configs = simplejson.load(f)
                config = configs["instances"]["synthetic"]

            with HighDimSynthesizer(df=data_missing, **config['params']) as synthesizer:
                synthesizer.learn(df_train=data_missing, num_iterations=n_iter)
                synthesized = synthesizer.synthesize(num_rows=len(data))

                max_ks_iter.append(max_ks_distance(data, synthesized))
                mean_ks_iter.append(mean_ks_distance(data, synthesized))
                max_corr_iter.append(max_correlation_distance(data, synthesized))
                mean_corr_iter.append(mean_correlation_distance(data, synthesized))

        print('Computed results for {}% NaNs for {}. Took {:.2f}s.'.format(100 - i, mechanism, time.time() - t_start))
        max_ks_vec.append(np.mean(max_ks_iter))
        mean_ks_vec.append(np.mean(mean_ks_iter))
        max_corr_vec.append(np.mean(max_corr_iter))
        mean_corr_vec.append(np.mean(mean_corr_iter))

    plt.figure(figsize=(12, 8))
    plt.plot(keep_ratio_vec, max_ks_vec, label='Max KS Distance')
    plt.plot(keep_ratio_vec, mean_ks_vec, label='Mean KS Distance')
    plt.plot(keep_ratio_vec, max_corr_vec, label='Max Correlation Distance')
    plt.plot(keep_ratio_vec, mean_corr_vec, label='Mean Correlation Distance')
    plt.legend()
    plt.title(mechanism)
    plt.xlabel('Non-Missing Ratio')
    plt.ylabel('Distance')
    plt.show()

    return max_ks_vec, mean_ks_vec, max_corr_vec, mean_corr_vec
