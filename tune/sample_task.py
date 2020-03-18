import pandas as pd
import ray
from ax.service.ax_client import AxClient
from ray import tune
from ray.tune import track
from ray.tune.suggest.ax import AxSearch

from synthesized import HighDimSynthesizer

loss_sample_size = 50_000
data = pd.read_csv('data/credit_with_categoricals.csv')
data = data.dropna()
loss_sample_size = min(loss_sample_size, len(data))

ax = AxClient(enforce_sequential_optimization=False)
ax.create_experiment(
    name="capacity_tuning",
    parameters=[
        {"name": "capacity", "type": "range", "bounds": [8, 128]},
        {"name": "num_layers", "type": "range", "bounds": [1, 4]},
        {"name": "residual_depths", "type": "range", "bounds": [2, 6]}
    ],
    objective_name="mean_loss",
    minimize=True
)


def train_evaluate(parameterization):
    print(parameterization)
    with HighDimSynthesizer(df=data, **parameterization) as synthesizer:
        synthesizer.learn(data, num_iterations=None)
        data_ = synthesizer.preprocess(data.sample(loss_sample_size))
        feed_dict = synthesizer.get_data_feed_dict(data_)
        losses = synthesizer.get_losses(data=feed_dict)
        loss = losses['kl-loss'] + losses['reconstruction-loss']
        loss = loss.numpy().item()
        track.log(
            mean_loss=loss,
        )


ray.init(address='auto', redis_password='5241590000000000')

tune.run(
    train_evaluate,
    num_samples=60,
    search_alg=AxSearch(ax, max_concurrent=20),  # Note that the argument here is the `AxClient`.
    verbose=0,  # Set this level to 1 to see status updates and to 2 to also see trial results.
    # To use GPU, specify: resources_per_trial={"gpu": 1}.
    resources_per_trial={"cpu": 2},
    max_failures=3
)

best_parameters, values = ax.get_best_parameters()

print('best_parameters', best_parameters)
print('values', values)
print(ax.get_trials_data_frame().sort_values('trial_index'))
