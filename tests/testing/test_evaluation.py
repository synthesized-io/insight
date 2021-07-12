import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from synthesized.testing.evaluation import (Evaluation, baseline_evaluation_and_plot, plot_data, plot_multidimensional,
                                            synthesize_and_plot)

matplotlib.use('Agg')


def test_synthesize_and_plot():
    x = np.random.normal(0, 1, 1000)
    y = x + np.random.normal(0, 0.5, 1000)
    df = pd.DataFrame({'x': x, 'y': y})

    eval = Evaluation('test', 'test', 'test', 'test')

    synthesize_and_plot(df, 'test', eval, {'params': {}, 'num_iterations': 100},
                        plot_losses=True, plot_distances=True,
                        show_distributions=True, show_distribution_distances=True,
                        show_emd_distances=True, show_correlation_distances=True,
                        show_correlation_matrix=True, show_cramers_v_distances=True,
                        show_cramers_v_matrix=True, show_logistic_rsquared_distances=True,
                        show_logistic_rsquared_matrix=True)

    baseline_evaluation_and_plot(df, 'test', eval)

    plot_data(df, plt.subplots(1,1)[1])
    plot_multidimensional(df.sample(500), df.sample(500))
