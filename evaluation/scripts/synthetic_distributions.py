import os
import json

from synthesized.testing.evaluation import Evaluation
from synthesized.testing import synthetic_distributions as syn_dist
from synthesized.testing import metrics as eval_metrics
from synthesized.testing import plotting as syn_plot


if not 'workbookDir' in globals():
    workbookDir = os.getcwd()
os.chdir(os.path.split(os.path.split(workbookDir)[0])[0])


branch = os.environ.get('evaluation_branch', 'n/a')
revision = os.environ.get('evaluation_revision', 'n/a')
group = "synthetic"
config_path = os.environ.get('evaluation_config_path', "configs/evaluation/synthetic_distributions.json")
with open(config_path, 'r') as f:
    configs = json.load(f)
    config = configs["instances"]["synthetic"]
evaluation = Evaluation(branch=branch, revision=revision, group=group, 
                        metrics_file="../highdim-exp/metrics-highdim-exp.jsonl")

# Gauss "ball" outside of center
data = syn_dist.create_gauss_ball(x_mean=1000, x_std=100, y_mean=100, y_std=10, size=10000)
_ = syn_plot.synthesize_and_plot(data, 'ball', evaluation=evaluation, metrics=eval_metrics.default_metrics, 
                                 config=config)

# Gauss "ball" around of zero
data = syn_dist.create_gauss_ball(x_mean=0, x_std=100, y_mean=0, y_std=10, size=10000)
_ = syn_plot.synthesize_and_plot(data, 'ball_ext', evaluation=evaluation, metrics=eval_metrics.default_metrics, 
                                 config=config)

# Correlated Gaussian far from zero
data = syn_dist.create_gauss_ball(x_mean=1000, x_std=100, y_mean=100, y_std=10, size=10000, cor=0.8)
metrics = dict(eval_metrics.default_metrics)
metrics["max_correlation_distance"] = eval_metrics.max_correlation_distance
_ = syn_plot.synthesize_and_plot(data, 'corr_ball_far', evaluation=evaluation, metrics=metrics, 
                                 config=config)

# Correlated Gaussian around zero
data = syn_dist.create_gauss_ball(x_mean=0, x_std=100, y_mean=0, y_std=10, size=10000, cor=0.8)
metrics = dict(eval_metrics.default_metrics)
metrics["max_correlation_distance"] = eval_metrics.max_correlation_distance
_ = syn_plot.synthesize_and_plot(data, 'corr_ball_zero', evaluation=evaluation, metrics=metrics, 
                                 config=config)

# Line of noise that far from zero
data = syn_dist.create_line(x_range=(0, 1000), intercept=100, slope=-0.1, y_std=10, size=10000)
_ = syn_plot.synthesize_and_plot(data, 'line', evaluation=evaluation, metrics=eval_metrics.default_metrics, 
                                 config=config)

# Line of noise that comes from zero
data = syn_dist.create_line(x_range=(0, 1000), intercept=0, slope=0.1, y_std=10, size=10000)
_ = syn_plot.synthesize_and_plot(data, 'line_ext', evaluation=evaluation, metrics=eval_metrics.default_metrics, 
                                 config=config)

# Power law distribution
data = syn_dist.create_power_law_distribution(shape=0.5, scale=1000, size=10000)
_ = syn_plot.synthesize_and_plot(data, 'power_law', evaluation=evaluation, metrics=eval_metrics.default_metrics, 
                                 config=config)

# Conditional distribution
data = syn_dist.create_conditional_distribution((10,2), (20, 5), (30, 1), size=10000)
_ = syn_plot.synthesize_and_plot(data, 'conditional', evaluation=evaluation, metrics=eval_metrics.default_metrics, 
                                 config=config)

# Bernoulli distribution
data = syn_dist.create_bernoulli(probability=0.5, size=10000)
_ = syn_plot.synthesize_and_plot(data, 'bernoulli_50/50', evaluation=evaluation, metrics=eval_metrics.default_metrics, 
                                 config=config)


data = syn_dist.create_bernoulli(probability=0.2, size=10000)
_ = syn_plot.synthesize_and_plot(data, 'bernoulli_20/80', evaluation=evaluation, metrics=eval_metrics.default_metrics, 
                                 config=config)

# Categorical distribution
data = syn_dist.create_uniform_categorical(n_classes=100, size=100000)
_ = syn_plot.synthesize_and_plot(data, 'categorical_uniform', evaluation=evaluation, 
                                 metrics=eval_metrics.default_metrics, 
                                 config=config)

data = syn_dist.create_power_law_categorical(n_classes=10, size=10000)
_ = syn_plot.synthesize_and_plot(data, 'categorical_powerlaw', evaluation=evaluation,
                                 metrics=eval_metrics.default_metrics, 
                                 config=config)

data = syn_dist.create_mixed_continuous_categorical(n_classes=10, size=10000)
_ = syn_plot.synthesize_and_plot(data, 'mixed_categorical_continuous', evaluation=evaluation,
                                 metrics=eval_metrics.default_metrics, config=config, show_anova=True)

data = syn_dist.create_correlated_categorical(n_classes=10, size=10000, sd=1.)
_ = syn_plot.synthesize_and_plot(data, 'correlated_categoricals', evaluation=evaluation,
                                 metrics=eval_metrics.default_metrics, config=config, show_cat_rsquared=True)

data = syn_dist.create_multidimensional_categorical(dimensions=50, n_classes=10, size=10000)
_ = syn_plot.synthesize_and_plot(data, 'multidimensional_categorical', evaluation=evaluation,
                                 metrics=eval_metrics.default_metrics, config=config)

data = syn_dist.create_multidimensional_correlated_categorical(dimensions=50, n_classes=10, size=10000)
_ = syn_plot.synthesize_and_plot(data, 'multidimensional_correlated_categorical', evaluation=evaluation,
                                 metrics=eval_metrics.default_metrics, config=config)

data = syn_dist.create_multidimensional_mixed(categorical_dim=25, continuous_dim=25, n_classes=10, size=10000)
_ = syn_plot.synthesize_and_plot(data, 'multidimensional_mixed', evaluation=evaluation,
                                 metrics=eval_metrics.default_metrics, config=config)

data = syn_dist.create_multidimensional_correlated_mixed(categorical_dim=25, continuous_dim=25, n_classes=10,
                                                         size=10000, categorical_sd=0.1, cont_sd=0.1, prior_sd=0.5)
_ = syn_plot.synthesize_and_plot(data, 'multidimensional_correlated_mixed', evaluation=evaluation,
                                 metrics=eval_metrics.default_metrics, config=config)

evaluation.write_metrics()
