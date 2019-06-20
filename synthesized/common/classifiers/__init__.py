from .basic import BasicClassifier
from .classifier import Classifier

classifier_modules = dict(
    basic=BasicClassifier
)

__all__ = ['Classifier', 'classifier_modules']
