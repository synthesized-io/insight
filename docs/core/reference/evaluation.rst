==========
Evaluation
==========

Synthesized contains a variety of methods that can be used to assess the quality and utility
of the generated synthetic data emperically and visually.

.. currentmodule:: synthesized.insight.metrics

.. rubric:: Univariate Metrics

.. autosummary::
   :toctree: _api/
   :template: class.rst

   KolmogorovSmirnovDistance
   EarthMoversDistance

.. rubric:: Multivariate Metrics

.. autosummary::
   :toctree: _api/
   :template: class.rst

   CramersV
   KendallTauCorrelation
   CategoricalLogisticR2
   SpearmanRhoCorrelation

.. rubric:: Modelling Metrics

.. autosummary::
   :toctree: _api/

   predictive_modelling_score
   predictive_modelling_comparison

.. currentmodule:: synthesized.testing

.. rubric:: Plotting & Analysis

.. autosummary::
   :toctree: _api/
   :template: class.rst

   UtilityTesting
