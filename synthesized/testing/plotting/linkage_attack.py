import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm


def plot_linkage_attack(df_la: pd.DataFrame, ax: plt.Axes = None, marker_size: float = 1.0):
    """Creates a bubble plot showing vulnerable groups from a linkage attack."""
    df_la['num_original'] = df_la['original_values'].map(lambda x: len(x) if isinstance(x, list) else 0)
    df_la['severity'] = (1 - df_la['k_dist']) * df_la['t_orig']
    df_la['marker_size'] = df_la['num_original'].map(lambda x: marker_size * 50 * (np.log(x + 1) + 1))

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    else:
        fig = ax.get_figure()

    ax.set_title("Linkage Attack")
    ax.set_ylabel("t-closeness (How signficant the information gain is.)")
    ax.set_xlabel("k-distance (How different the information is.)")
    ax.scatter(
        x=df_la['k_dist'], y=df_la['t_attack'],
        s=df_la['marker_size'], c=df_la['severity'],
        alpha=0.25, cmap=cm.get_cmap('magma', 256)
    )
    return fig
