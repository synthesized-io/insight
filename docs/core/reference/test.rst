=========
Synthesis
=========

.. rubric:: Synthesizers

``Synthesizer`` objects are the core component of Synthesized. They provide implementations of the models that can
learn from data to produce highly accurate and representative synthetic data.

.. currentmodule:: synthesized

.. autosummary::
    :template: class.rst
    :toctree:
    :recursive:

    complex.HighDimSynthesizer
    complex.TwoTableSynthesizer

.. rubric:: Data Reshaping

In addition to generating synthetic data that looks and behaves exactly like the original, Synthesized provides
tools that enable data reshaping and generation of arbitrary scenarios.

.. autosummary::
    :toctree:
    :recursive:

    complex.ConditionalSampler
    complex.DataImputer