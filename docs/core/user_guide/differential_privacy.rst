Differential Privacy
====================

Synthesized can incorporate differential privacy techniques into the process that is used to build the generative model
of the data. This ensures a mathematical guarantee of an individual's privacy in the original data and reduces the
the risk of data leakage in the Synthesized data.

Specifically, Synthesized utilises :math:`(\epsilon, \delta)`-differential privacy defined by

.. math::

    \frac{\mathrm{P}(\mathcal{M}(D))}{\mathrm{P}(\mathcal{M}(D'))} \leq e^{\epsilon} + \delta

where :math:`D` is the original data, :math:`D'` is the original data with an individual's row removed, :math:`\mathcal{M}`
is a stochastic function of the original data, and :math:`\mathrm{P}` is the corresponding probability distribution.
Differential privacy provides an upper-bound on the privacy loss of an individual, by ensuring that the
output of the stochastic function :math:`\mathcal{M}` differs by at most :math:`e^{\epsilon}` when applied to a dataset
with the invidivual, and a dataset excluding the individual. The parameter :math:`\delta` relaxes the constraint,
allowing differential privacy to be used in a to a greater range of applications.

.. note::
    There is an unavoidable trade-off between the privacy and utility of a dataset. Smaller values of :math:`\epsilon`
    provide stronger privacy guarantees, but will inherently reduce the usefulness of the data.

Differential privacy can be enabled by setting by setting ``differential_privacy = True`` in
:class:`~synthesized.config.HighDimConfig`:

.. ipython:: python

    from synthesized.config import HighDimConfig
    config = HighDimConfig()
    config.differential_privacy = True

In addition, there are several parameters that will need to be adjusted depending on the desired level of privacy
as well as the size of dataset being learned.

- ``config.epsilon``: This sets the desired level of :math:`\epsilon`, with smaller values producing more private data.
  Training of the model is aborted if this value is reached. It is important to note that it may not be possible to
  obtain the desired level of :math:`\epsilon` as it depends strongly on the size of the dataset together with the
  amount of noise added.
- ``config.noise_multiplier``: The amont of noise added to ensure differential privacy can be achieved. Values are typicaly
  in the range `1.0` to `10.0`. Higher values allow smaller values of :math:`\epsilon` to be reached and therefore
  greater privacy, but lower data quality.

``config.l2_norm_clip``, ``config.num_microbatches``, and ``config.delta`` can also be tuned, but it is recommended to keep
them at their default value.

The :class:`~synthesized.config.HighDimConfig` can then be passed to :class:`~synthesized.complex.HighDimSynthesizer` to
train a Synthesizer with differential privacy guarantees.

.. ipython:: python
    :verbatim:

    synthesizer = HighDimSynthesizer(df_meta, config=config)
    synthesizer.learn(...)

.. warning::
    Enabling differential privacy may significantly slow down the learning process.

Once trained, the value of :math:`\epsilon` reached by the Synthesizer for the particular dataset can be obtained
with:

.. ipython:: python
    :verbatim:

    synthesizer.epsilon
    0.8

.. Note that due to the method used to learn the Synthesizer model, it is not possible to directly choose the desired
.. :math:`\epsilon`. However, if the model is trained for a fixed number of iterations, :math:`\epsilon` can be calculated using the
.. ``common.util.get_privacy_budget`` function.

.. .. ipython:: python

..     from synthesized.common.util import get_privacy_budget

..     epsilon = get_privacy_budget(noise_multiplier=1.2, steps=1000, batch_size=128, data_size=10000)
..     epsilon
