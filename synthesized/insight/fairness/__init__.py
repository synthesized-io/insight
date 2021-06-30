from synthesized.licence import LicenceError, OptionalFeature, verify

if verify(OptionalFeature.FAIRNESS):

    from .bias_mitigator import BiasMitigator
    from .fairness_scorer import FairnessScorer

    __all__ = ['FairnessScorer', 'BiasMitigator']

else:
    raise LicenceError('Please upgrade your licence to use fairness and bias features.')
