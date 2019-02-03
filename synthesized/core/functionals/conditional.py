import numpy as np
import tensorflow as tf

from .functional import Functional


class ConditionalFunctional(Functional):

    def __init__(
        self, comparison, reference, true_functional, false_functional=None, comparison_value=None,
        conditioned_values=None, values=None, name=None
    ):
        if values is None:
            assert comparison_value is not None
            assert conditioned_values is not None and len(conditioned_values) >= 1
            values = (comparison_value,) + tuple(conditioned_values)
        else:
            assert comparison_value is None and conditioned_values is None and len(values) >= 2
            comparison_value = values[0]
            conditioned_values = tuple(values[1:])

        super().__init__(values=values, name=name)

        self.comparison_value = comparison_value
        self.comparison = comparison
        self.reference = reference
        from . import functional_modules
        self.true_functional = self.add_module(
            module=true_functional, modules=functional_modules, values=conditioned_values
        )
        self.false_functional = self.add_module(
            module=false_functional, modules=functional_modules, values=conditioned_values
        )

    def specification(self):
        spec = super().specification()
        spec.update(
            comparison=self.comparison, reference=self.reference,
            true_functional=self.true_functional.specification(),
            false_functional=(
                None if self.false_functional is None else self.false_functional.specification()
            )
        )
        return spec

    def tf_loss(self, comparison_samples, *conditioned_samples):
        comparison = getattr(tf, self.comparison)
        comparison_mask = comparison(x=comparison_samples, y=self.reference)

        true_samples = [
            tf.boolean_mask(tensor=samples, mask=comparison_mask) for samples in conditioned_samples
        ]
        true_loss = self.true_functional.loss(*true_samples)

        # tf.count_nonzero()  scaled?

        if self.false_functional is None:
            return true_loss

        else:
            false_samples = [
                tf.boolean_mask(tensor=samples, mask=tf.logical_not(x=comparison_mask))
                for samples in conditioned_samples
            ]
            false_loss = self.false_functional.loss(*false_samples)

            return true_loss + false_loss

    def check_distance(self, comparison_samples, *conditioned_samples):
        comparison = getattr(np, self.comparison)
        comparison_mask = comparison(comparison_samples, self.reference)

        true_samples = [samples[comparison_mask] for samples in conditioned_samples]
        true_distance = self.true_functional.check_distance(*true_samples)

        if self.false_functional is None:
            return true_distance

        else:
            false_samples = [
                samples[np.logical_not(comparison_mask)] for samples in conditioned_samples
            ]
            false_distance = self.false_functional.check_distance(*false_samples)

            return true_distance + false_distance