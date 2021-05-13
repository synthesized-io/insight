import matplotlib
import numpy as np
import pandas as pd

from synthesized.testing.evaluation import Evaluation, synthesize_and_plot

matplotlib.use('Agg')


def test_synthesize_and_plot():
    x = np.random.normal(0, 1, 1000)
    y = x + np.random.normal(0, 0.5, 1000)
    df = pd.DataFrame({'x': x, 'y': y})

    eval = Evaluation('test', 'test', 'test', 'test')

    synthesize_and_plot(df, 'test', eval, {'params': {}, 'num_iterations': 100}, show_distribution_distances=True,
                        show_emd_distances=True, show_correlation_distances=True,
                        show_correlation_matrix=True, show_cramers_v_distances=True,
                        show_cramers_v_matrix=True, show_logistic_rsquared_distances=True,
                        show_logistic_rsquared_matrix=True)
