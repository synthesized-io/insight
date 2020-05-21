import os
import json

if not 'workbookDir' in globals():
    workbookDir = os.getcwd()
os.chdir(os.path.split(os.path.split(workbookDir)[0])[0])

from synthesized.insight.metrics import avg_kolmogorov_smirnov_distance, max_diff_kendell_tau_correlation
from synthesized.testing.evaluation import Evaluation
from synthesized.testing import synthetic_distributions as syn_dist
from synthesized.testing.evaluation import synthesize_and_plot as syn_plot


default_metrics = [avg_kolmogorov_smirnov_distance]

branch = os.environ.get('evaluation_branch', 'master')
revision = os.environ.get('evaluation_revision', 'a1b2c3d')
group = "synthetic"
config_path = os.environ.get('evaluation_config_path', "configs/evaluation/highdim-exp_synthetic_distributions.json")
with open(config_path, 'r') as f:
    configs = json.load(f)
    config = configs["instances"]["synthetic"]
evaluation = Evaluation(branch=branch, revision=revision, group=group, 
                        metrics_file="../highdim-exp/metrics-highdim-exp.jsonl")

# Gauss "ball" outside of center
data = syn_dist.create_gauss_ball(x_mean=1000, x_std=100, y_mean=100, y_std=10, size=10000)
_ = syn_plot(data, 'ball', evaluation=evaluation, eval_metrics=default_metrics, config=config)

# Gauss "ball" around of zero
data = syn_dist.create_gauss_ball(x_mean=0, x_std=100, y_mean=0, y_std=10, size=10000)
_ = syn_plot(data, 'ball_ext', evaluation=evaluation, eval_metrics=default_metrics, config=config)

# Correlated Gaussian far from zero
metrics = default_metrics.copy()
metrics.append(max_diff_kendell_tau_correlation)
data = syn_dist.create_gauss_ball(x_mean=1000, x_std=100, y_mean=100, y_std=10, size=10000, cor=0.8)
_ = syn_plot(data, 'corr_ball_far', evaluation=evaluation, eval_metrics=metrics, config=config)

# Correlated Gaussian around zero
data = syn_dist.create_gauss_ball(x_mean=0, x_std=100, y_mean=0, y_std=10, size=10000, cor=0.8)
_ = syn_plot(data, 'corr_ball_zero', evaluation=evaluation, eval_metrics=metrics, config=config)

# Line of noise that far from zero
data = syn_dist.create_line(x_range=(0, 1000), intercept=100, slope=-0.1, y_std=10, size=10000)
_ = syn_plot(data, 'line', evaluation=evaluation, eval_metrics=default_metrics, config=config)

# Line of noise that comes from zero
data = syn_dist.create_line(x_range=(0, 1000), intercept=0, slope=0.1, y_std=10, size=10000)
_ = syn_plot(data, 'line_ext', evaluation=evaluation, eval_metrics=default_metrics, config=config)

# Power law distribution
data = syn_dist.create_power_law_distribution(shape=0.5, scale=1000, size=10000)
_ = syn_plot(data, 'power_law', evaluation=evaluation, eval_metrics=default_metrics, config=config)

# Conditional distribution
data = syn_dist.create_conditional_distribution((10,2), (20, 5), (30, 1), size=10000)
_ = syn_plot(data, 'conditional', evaluation=evaluation, eval_metrics=default_metrics, config=config)

# Bernoulli distribution
data = syn_dist.create_bernoulli(probability=0.5, size=10000)
_ = syn_plot(data, 'bernoulli_50/50', evaluation=evaluation, eval_metrics=default_metrics, config=config)


data = syn_dist.create_bernoulli(probability=0.2, size=10000)
_ = syn_plot(data, 'bernoulli_20/80', evaluation=evaluation, eval_metrics=default_metrics, config=config)

# Categorical distribution
data = syn_dist.create_uniform_categorical(n_classes=100, size=100000)
_ = syn_plot(data, 'categorical_uniform', evaluation=evaluation, eval_metrics=default_metrics, config=config)

data = syn_dist.create_power_law_categorical(n_classes=10, size=10000)
_ = syn_plot(data, 'categorical_powerlaw', evaluation=evaluation, eval_metrics=default_metrics, config=config)

data = syn_dist.create_mixed_continuous_categorical(n_classes=10, size=10000)
_ = syn_plot(data, 'mixed_categorical_continuous', evaluation=evaluation, eval_metrics=default_metrics, config=config,
             show_cat_rsquared=True)

data = syn_dist.create_correlated_categorical(n_classes=10, size=10000, sd=1.)
_ = syn_plot(data, 'correlated_categoricals', evaluation=evaluation, eval_metrics=default_metrics, config=config,
             show_cramers_v_matrix=True, show_cramers_v_distances=True)

data = syn_dist.create_multidimensional_categorical(dimensions=50, n_classes=10, size=10000)
_ = syn_plot(data, 'multidimensional_categorical', evaluation=evaluation, eval_metrics=default_metrics, config=config)

data = syn_dist.create_multidimensional_correlated_categorical(dimensions=50, n_classes=10, size=10000)
_ = syn_plot(data, 'multidimensional_correlated_categorical', evaluation=evaluation, eval_metrics=default_metrics,
             config=config)

data = syn_dist.create_multidimensional_mixed(categorical_dim=25, continuous_dim=25, n_classes=10, size=10000)
_ = syn_plot(data, 'multidimensional_mixed', evaluation=evaluation, eval_metrics=default_metrics, config=config)

data = syn_dist.create_multidimensional_correlated_mixed(categorical_dim=25, continuous_dim=25, n_classes=10,
                                                         size=10000, categorical_sd=0.1, cont_sd=0.1, prior_sd=0.5)
_ = syn_plot(data, 'multidimensional_correlated_mixed', evaluation=evaluation, eval_metrics=default_metrics,
             config=config)

evaluation.write_metrics()
