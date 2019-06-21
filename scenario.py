import argparse
from datetime import datetime
import json
import os

from synthesized.scenario import ScenarioSynthesizer


print()
print(datetime.now().strftime('%H:%M:%S'), 'Parse arguments...', flush=True)
parser = argparse.ArgumentParser()
parser.add_argument(
    '-s', '--scenario', type=str, default='scenarios/example.json', help="scenario"
)
parser.add_argument('-n', '--num-iterations', type=int, default=100, help="training iterations")
parser.add_argument('-m', '--num-samples', type=int, default=1024, help="training samples")
parser.add_argument(
    '-y', '--hyperparameters', default='capacity=8', help="list of hyperparameters (comma, equal)"
)
parser.add_argument('-b', '--tensorboard', type=str, default=None, help="TensorBoard summaries")
args = parser.parse_args()
print()


print(datetime.now().strftime('%H:%M:%S'), 'Load scenario...', flush=True)
if os.path.isfile(args.scenario):
    filename = args.scenario
elif os.path.isfile(os.path.join('scenarios', args.scenario)):
    filename = os.path.join('scenarios', args.scenario)
elif os.path.isfile(os.path.join('scenarios', args.scenario + '.json')):
    filename = os.path.join('scenarios', args.scenario + '.json')
else:
    assert False
with open(filename, 'r') as filehandle:
    scenario = json.load(fp=filehandle)
assert len(scenario) == 2 and 'functionals' in scenario and 'values' in scenario


print(datetime.now().strftime('%H:%M:%S'), 'Initialize synthesizer...', flush=True)
if args.hyperparameters is None:
    synthesizer = ScenarioSynthesizer(
        values=scenario['values'], functionals=scenario['functionals'], summarizer=args.tensorboard
    )
else:
    kwargs = [kv.split('=') for kv in args.hyperparameters.split(',')]
    kwargs = {key: float(value) if '.' in value else int(value) for key, value in kwargs}
    synthesizer = ScenarioSynthesizer(
        values=scenario['values'], functionals=scenario['functionals'],
        summarizer=args.tensorboard, **kwargs
    )
print(repr(synthesizer))
print()


print(datetime.now().strftime('%H:%M:%S'), 'Value types...', flush=True)
for value in synthesizer.values:
    print(value.name, value)
print()


print(datetime.now().strftime('%H:%M:%S'), 'Synthesis...', flush=True)
with synthesizer:
    print(datetime.now().strftime('%H:%M:%S'), 'Start learning...', flush=True)
    synthesizer.learn(
        num_iterations=args.num_iterations, num_samples=args.num_samples, callback_freq=20
    )
    print(datetime.now().strftime('%H:%M:%S'), 'Finished learning...', flush=True)
    synthesized = synthesizer.synthesize(num_rows=10000)
    assert len(synthesized) == 10000
    for functional in synthesizer.functionals:
        if functional.required_outputs() == '*':
            samples_args = synthesized
        else:
            samples_args = [synthesized[label] for label in functional.required_outputs()]
        print(functional.name, functional.check_distance(*samples_args), flush=True)
print()


print(datetime.now().strftime('%H:%M:%S'), 'Synthesized data...', flush=True)
print(synthesized.head(5))
print()
