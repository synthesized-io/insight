import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.metrics.scorer import roc_auc_scorer
from sklearn.model_selection import train_test_split

from synthesized.core import BasicSynthesizer
from synthesized.core.classifiers import BasicClassifier
from synthesized.tuning import HyperparamSpec


NUM_RUNS = 3


print()
print('Parse arguments...')
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, help="dataset name")
parser.add_argument('-t', '--target', default=-1, help="target column")
parser.add_argument(
    '-y', '--hyperparams', type=str, default='default', help="hyperparameter specification"
)
parser.add_argument('-s', '--score', type=str, default='sklearn', help="score")
parser.add_argument(
    '-v', '--values', default='random', help="random, or number of values for grid search"
)
parser.add_argument(
    '-i', '--iterations', type=int, default=2000, help="number of iterations per step"
)
parser.add_argument(
    '-o', '--tolerance', type=int, default=3, help="number of deteriorating steps to tolerate"
)
parser.add_argument(
    '-m', '--max-iterations', type=int, default=100000, help="max number of iterations"
)
parser.add_argument(
    '-c', '--classifier-iterations', type=int, default=1000, help="classifier training iterations"
)
# parser.add_argument('--tfrecords', action='store_true', help="from TensorFlow records")
args = parser.parse_args()


print('Load dataset...')
data = pd.read_csv('data/{}.csv'.format(args.dataset))
# tfrecords_filename = 'data/{}.tfrecords'.format(args.dataset)
num_with_nan = len(data)
data = data.dropna()
print('Nans dropped:', num_with_nan - len(data), 'of', num_with_nan)
if isinstance(args.target, int):
    target = data.columns[args.target]
elif args.target.isdigit():
    target = data.columns[int(args.target)]
else:
    target = args.target
print('Target column:', target)
print()


print('Prepare data...')
data.sort_index(inplace=True)
original, heldout = train_test_split(data, test_size=0.2)
original.reset_index(drop=True, inplace=True)
heldout.reset_index(drop=True, inplace=True)
print()


print('Load hyperparameter specification...')
hyperparam_spec = HyperparamSpec(specification='configs/tuning/{}.json'.format(args.hyperparams))
if args.values == 'random':
    iterator = hyperparam_spec.random()
    print('random search')
elif ',' in args.values:
    iterator = hyperparam_spec.grid(
        num_values=[int(num_value) for num_value in args.values.split(',')]
    )
    print('grid search:', args.values)
else:
    iterator = hyperparam_spec.grid(num_values=int(args.values))
    print('grid search:', args.values)
best_scores = [None for _ in range(10)]
best_hyperparams = [None for _ in range(10)]
print()


def neg_ks_distance_score(synthesizer, synthesized):
    synthesized = synthesizer.preprocess(data=synthesized)
    data_ = synthesizer.preprocess(data.copy())
    return -np.mean([ks_2samp(data_[col], synthesized[col])[0] for col in synthesized.columns])


def sklearn_score(synthesizer, synthesized):
    train = synthesizer.preprocess(data=synthesized)
    estimator = GradientBoostingClassifier()
    try:
        estimator.fit(X=train.drop(labels=target, axis=1), y=train[target])
        test = synthesizer.preprocess(data=heldout.copy())
        return roc_auc_scorer(
            clf=estimator, X=test.drop(labels=target, axis=1), y=test[target]
        )
    except ValueError:
        return None


def tf_score(synthesizer, synthesized):
    with BasicClassifier(data=data, target_label=target) as classifier:
        classifier.learn(num_iterations=args.classifier_iterations, data=synthesized)
        classified = classifier.classify(data=heldout.drop(labels=target, axis=1))
    return f1_score(y_true=heldout[target], y_pred=classified[target], average='macro')


def pessimistic_score(hyperparams, n_runs, scorer_fn):
    def _score():
        history = [None for _ in range(args.tolerance)]
        with BasicSynthesizer(data=data, exclude_encoding_loss=True, **hyperparams) as synthesizer:
            print(datetime.now().strftime('%H:%M:%S'), 'training...', flush=True)
            for _ in range(args.max_iterations // args.iterations):
                synthesizer.learn(num_iterations=args.iterations, data=original.copy())
                synthesized = synthesizer.synthesize(n=10000)
                print(datetime.now().strftime('%H:%M:%S'), 'computing score...', flush=True)
                result = scorer_fn(synthesizer, synthesized)
                print(datetime.now().strftime('%H:%M:%S'), 'synthetic score:', result, flush=True)
                print()
                history.pop(0)
                history.append(result)
                if any(score is None for score in history):
                    continue
                if all(history[n] < history[0] for n in range(1, len(history))):
                    break
            return max(history)
    return min([_score() for _ in range(n_runs)])


for iteration, hyperparams in enumerate(iterator):
    print('==============================')
    print('        iteration {:>4}        '.format(iteration))
    print('==============================')
    print()
    print('params:', ', '.join('{}={}'.format(*h) for h in hyperparams.items()))
    print()

    if args.score == 'neg_ks_distance':
        score = pessimistic_score(hyperparams, NUM_RUNS, neg_ks_distance_score)
        print('neg KS-distance:', score)
        print()

    elif args.score == 'sklearn':
        score = pessimistic_score(hyperparams, NUM_RUNS, sklearn_score)
        print('sklearn (roc_auc):', score)
        print()

    elif args.score == 'tf':
        score = pessimistic_score(hyperparams, NUM_RUNS, tf_score)
        print('tf (f1):', score)
        print()

    else:
        raise NotImplementedError

    print('Update best hyperparameters...')
    for n, best_score in enumerate(best_scores):
        if best_score is None or score > best_score:
            best_scores[n + 1:] = best_scores[n: -1]
            best_scores[n] = score
            best_hyperparams[n + 1:] = best_hyperparams[n: -1]
            best_hyperparams[n] = hyperparams
            break
    for n in range(10):
        if best_hyperparams[n] is None:
            break
        str_hyperparams = ','.join(
            '{}={}'.format(*hyparparam) for hyparparam in best_hyperparams[n].items()
        )
        print('{}: {} ({})'.format(n + 1, best_scores[n], str_hyperparams))
    print()
