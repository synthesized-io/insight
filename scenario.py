import argparse
from datetime import datetime
import json
from synthesized.common import ScenarioSynthesizer


print()
print('Parse arguments...')
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--iterations', type=int, help="training iterations")
parser.add_argument('-n', '--num-samples', type=int, default=1024, help="training samples")
parser.add_argument('-e', '--evaluation', type=int, default=0, help="evaluation frequency")
parser.add_argument(
    '-s', '--scenario', type=str, default='configs/scenarios/example.json', help="training samples"
)
parser.add_argument(
    '-y', '--hyperparameters', default=None, help="list of hyperparameters (comma, equal)"
)
parser.add_argument('-b', '--tensorboard', action='store_true', help="TensorBoard summaries")
args = parser.parse_args()
if args.evaluation == 0:
    args.evaluation = args.iterations


print('Initialize synthesizer...')
with open(args.scenario, 'r') as filehandle:
    scenario = json.load(fp=filehandle)
assert len(scenario) == 2 and 'functionals' in scenario and 'values' in scenario
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


print('Synthesis...')
with synthesizer:
    print(datetime.now().strftime('%H:%M:%S'), flush=True)
    for i in range(args.iterations // args.evaluation):
        synthesizer.learn(num_iterations=args.evaluation, num_samples=args.num_samples)
        # print(datetime.now().strftime('%H:%M:%S'), flush=True)
        # synthesized = synthesizer.synthesize(n=100000)
        # evaluation?
        print(datetime.now().strftime('%H:%M:%S'), i * args.evaluation, flush=True)
        print('Distances...')
        synthesized = synthesizer.synthesize(n=10000)
        for functional in synthesizer.functionals:
            if functional.required_outputs() == '*':
                samples_args = synthesized
            else:
                samples_args = [synthesized[label] for label in functional.required_outputs()]
            print(functional.name, functional.check_distance(*samples_args), flush=True)
        print()

print('Synthetic data...')
print(synthesized.head(20))
print()
