import argparse
from datetime import datetime
import sys
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from synthesized.core import BasicSynthesizer
from synthesized.core.classifiers import BasicClassifier


print('Parse arguments...')
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, help="dataset name")
parser.add_argument('-t', '--target', default=-1, help="target column")
parser.add_argument('-i', '--iterations', type=int, default=10000, help="training iterations")
parser.add_argument('--tfrecords', action='store_true', help="from TensorFlow records")
args = parser.parse_args()


print('Load dataset...')
data = pd.read_csv('data/{}.csv'.format(args.dataset))
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


print('Original data...')
print(original.head(5))
print()


print('Initialize synthesizer...')
synthesizer = BasicSynthesizer(data=data, iterations=args.iterations)
print(repr(synthesizer))
print()


print('Most frequent score...')
train = synthesizer.preprocess(data=original.copy())
estimator = DummyClassifier(strategy='most_frequent')
estimator.fit(X=train.drop(labels=target, axis=1), y=train[target])
test = synthesizer.preprocess(data=heldout.copy())
score = estimator.score(X=test.drop(labels=target, axis=1), y=test[target])
print('most frequent:', score)
print()


print('Original score...')
estimator = LogisticRegression()
estimator.fit(X=train.drop(labels=target, axis=1), y=train[target])
predictions = estimator.predict(X=test.drop(labels=target, axis=1))
accuracy = accuracy_score(y_true=test[target], y_pred=predictions)
precision = precision_score(y_true=test[target], y_pred=predictions)
recall = recall_score(y_true=test[target], y_pred=predictions)
f1 = f1_score(y_true=test[target], y_pred=predictions)
print('original:', accuracy, precision, recall, f1)
print()


print('Original classifier score...')
with BasicClassifier(data=data, target_label=target, iterations=args.iterations) as classifier:
    print(datetime.now().strftime('%H:%M:%S'), flush=True)
    if args.tfrecords:
        classifier.learn(filenames=('data/{}.tfrecords'.format(args.dataset),), verbose=10000)
    else:
        classifier.learn(data=original.copy(), verbose=10000)
    print(datetime.now().strftime('%H:%M:%S'), flush=True)
    classified = classifier.classify(data=heldout.drop(labels=target, axis=1))
    print(datetime.now().strftime('%H:%M:%S'), flush=True)
    print()
# score = (classified[target] == heldout[target]).sum() / len(classified)
accuracy = accuracy_score(y_true=heldout[target], y_pred=classified[target])
precision = precision_score(y_true=heldout[target], y_pred=classified[target])
recall = recall_score(y_true=heldout[target], y_pred=classified[target])
f1 = f1_score(y_true=heldout[target], y_pred=classified[target])
print('original classifier:', accuracy, precision, recall, f1)
print()


print('Synthesis...')
with synthesizer:
    print(datetime.now().strftime('%H:%M:%S'), flush=True)
    if args.tfrecords:
        synthesizer.learn(filenames=('data/{}.tfrecords'.format(args.dataset),), verbose=10000)
    else:
        synthesizer.learn(data=original.copy(), verbose=10000)
    print(datetime.now().strftime('%H:%M:%S'), flush=True)
    synthesized = synthesizer.synthesize(n=10000)
    print(datetime.now().strftime('%H:%M:%S'), flush=True)
    print()


print('Synthetic data...')
print(synthesized.head(10))
print()


print('Synthetic score...')
train = synthesizer.preprocess(data=synthesized.copy())
estimator = LogisticRegression()
estimator.fit(X=train.drop(labels=target, axis=1), y=train[target])
test = synthesizer.preprocess(data=heldout.copy())
predictions = estimator.predict(X=test.drop(labels=target, axis=1))
accuracy = accuracy_score(y_true=test[target], y_pred=predictions)
precision = precision_score(y_true=test[target], y_pred=predictions)
recall = recall_score(y_true=test[target], y_pred=predictions)
f1 = f1_score(y_true=test[target], y_pred=predictions)
print('synthesized:', accuracy, precision, recall, f1)
print()


print('Synthetic classifier score...')
with BasicClassifier(data=data, target_label=target, iterations=args.iterations) as classifier:
    print(datetime.now().strftime('%H:%M:%S'), flush=True)
    if args.tfrecords:
        classifier.learn(filenames=('data/{}.tfrecords'.format(args.dataset),), verbose=10000)
    else:
        classifier.learn(data=synthesized.copy(), verbose=10000)
    print(datetime.now().strftime('%H:%M:%S'), flush=True)
    classified = classifier.classify(data=heldout.drop(labels=target, axis=1))
    print(datetime.now().strftime('%H:%M:%S'), flush=True)
    print()
# score = (classified[target] == heldout[target]).sum() / len(classified)
accuracy = accuracy_score(y_true=heldout[target], y_pred=classified[target])
precision = precision_score(y_true=heldout[target], y_pred=classified[target])
recall = recall_score(y_true=heldout[target], y_pred=classified[target])
f1 = f1_score(y_true=heldout[target], y_pred=classified[target])
print('synthesized classifier:', accuracy, precision, recall, f1)
print()
