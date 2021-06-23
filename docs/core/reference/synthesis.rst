=========
Synthesis
=========

.. rubric:: Synthesizers

``Synthesizer`` objects are the core component of Synthesized. They provide implementations of the models that can
learn from data to produce highly accurate and representative synthetic data.

.. currentmodule:: synthesized.complex

.. autosummary::
    :template: class.rst
    :toctree: _api/

    HighDimSynthesizer
    TwoTableSynthesizer

.. rubric:: Data Reshaping & Augmentation

In addition to generating synthetic data that looks and behaves exactly like the original, Synthesized provides
tools that enable data reshaping and generation of arbitrary scenarios.

.. autosummary::
    :template: class.rst
    :toctree: _api/

    ConditionalSampler
    DataImputer


.. rubric:: Rules

Business logic and constraints can be specified using rule classes, and used within
:meth:`HighDimSynthesizer.synthesize_from_rules`.

.. currentmodule:: synthesized.common.rules

.. autosummary::
   :template: class.rst
   :toctree: _api/

   Association
   Expression
   GenericRule
   ValueIsIn
   ValueEquals
   ValueRange
   CaseWhenThen
