.. _conditions_guide:

================
Data Rebalancing
================

The basic use of ``HighDimSynthesizer`` is to generate a Synthesized version of a dataset where the statistical
properties of both are as close as possible, as described in the ref:`single table synthesis guide.<singletable_guide>`.
With ``ConditionalSampler`` the user has the capability of generating a new dataset with user-defined marginal
distributions, while still keeping the rest of the statistical properties as close as possible to those in the original
dataset.

This *conditional sampling* technique can be used in many situations, for example, to upsample rare events and
improve a model's predictive performance in highly imbalanced datasets, or to generate custom scenarios to validate
proper system behaviour.

.. note::
    A more extended analysis on different applications of data rebalancing and augmentation using Synthesized
    can be obtained from our webpage:
    `Data Science Applications of the Synthesized Platform`_

.. _Data Science Applications of the Synthesized Platform: https://www.synthesized.io/reports-and-whitepapers/data-science-applications-of-the-synthesized-platform

Similarly, rules can be defined in Synthesized to generate data that corresponds to a custom scenario: :ref:`See the
rules user guide<rules_guide>`.

Conditional Sampling
^^^^^^^^^^^^^^^^^^^^

.. ipython:: python

    from synthesized import ConditionalSampler

Conditional sampling is achieved using the ``ConditionalSampler`` class. This requires a ``HighDimSynthesizer`` instance
that has already been learned on the desired dataset.

.. ipython:: python
    :verbatim:

    sampler = ConditionalSampler(synthesizer) # synthesizer is an HighDimSynthesizer instance

The desired conditions to generate are specified by a normalized marginal distribution for the desired columns. For
example, consider a transaction dataset with a categorical ``transaction_flag`` field that has two categories:
``fraud``, ``not-fraud``, which contains only 5% of ``fraud`` transactions. A machine learning model trained on a
dataset like this could lead to unexpected results if the target imbalance is not treated carefully.

This problem could be easily solve by upsampling the minority class to obtain a new dataset with 50% ``fraud`` and
50% ``not fraud`` samples. To do so with ``ConditionalSampler``, the desired marginal
distribution can be specified as a `dict` with specification ``{category[str]: normalised_probablity[float]}``

.. ipython:: python

    fraud_marginal = {"fraud": 0.5, "not-fraud": 0.5}

And then generate the new dataset with the previously initialized ``ConditionalSampler``:

.. ipython:: python
    :verbatim:

    sampler.synthesize(
        num_rows=100,
        explicit_marginals={'transaction_flag': fraud_marginal}
    )

.. note::
    It is also possible to generate only ``fraud`` transactions. The marginal specification would then
    be ``fraud_marginal = {"fraud": 1.0, "not-fraud": 0.0}``.

To specify conditions for continuous fields, the categories must be given as bin edges of the form e.g ``"[low, high)"``
, e.g

.. ipython:: python
    :verbatim:

    age_marginal = {'[0.0, 50.0)': 0.3, '[50.0, 100.0)': 0.7}
    sampler.synthesize(
        num_rows=100,
        explicit_marginals={'age': age_marginal}
    )

Furthermore, the user can conditionally sample the new dataset on multiple marginal distributions, for example:

.. ipython:: python
    :verbatim:

    sampler.synthesize(
        num_rows=100,
        explicit_marginals={
            'transaction_flag': transaction_marginal,
            'age': age_marginal
        }
    )

.. warning::
    It's important to correctly define the ``explicit_marginals`` argument, otherwise the ``ConditionalSampler`` will
    raise a ``ValueError``. This dictionary must contain a dictionary ``Dict[column_name, marginal]``, where marginal
    has the format ``Dict[str, str]``, and the keys are a string containing the category/interval name, and the 
    values contain the probability of that category/interval.

    Additionally, all values in a ``marginal`` must add up to 1.

Alter Distributions
^^^^^^^^^^^^^^^^^^^

With *conditional sampling*, the output dataset is fully synthetic and doesn't contain any sample from the original
dataset. But it is also possible to alter the distributions of a given dataset, and obtain a new dataset with a
specific size, desired marginal distributions, and that contains a mix of Synthesized and original data.

This is achieved with the ``ConditionalSampler.alter_distributions()`` method:

.. ipython:: python
    :verbatim:

    sampler.alter_distributions(
        df=df_original,
        num_rows=1000,
        explicit_marginals={
            'transaction_flag': transaction_marginal,
            'age': age_marginal
        }
    )
