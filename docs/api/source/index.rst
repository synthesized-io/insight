.. synthesized documentation master file, created by
   sphinx-quickstart on Tue Aug 18 14:12:39 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Synthesized
===========

Synthesized SDK 1.0.0 Documentation
***********************************

Synthesizers
------------

..  autoclass:: synthesized.api.Synthesizer
    :members:

.. autoclass:: synthesized.api.HighDimSynthesizer
    :members:


MetaData
--------

.. autoclass:: synthesized.api.DataFrameMeta
   :members:

.. autoclass:: synthesized.api.MetaExtractor
   :members:

.. autoclass:: synthesized.api.TypeOverride
   :members:


Insight
-------

.. automodule:: synthesized.api.latent
   :members: get_latent_space, get_data_quality, latent_dimension_usage, total_latent_space_usage

.. automodule:: synthesized.api.modelling
   :members: predictive_modelling_score, predictive_modelling_comparison


Binaries
--------

.. autoclass:: synthesized.api.Binary
   :members:

.. autoclass:: synthesized.api.ModelBinary
   :members:

.. autoclass:: synthesized.api.DatasetBinary
   :members:

.. autoclass:: synthesized.api.BinaryType
   :members:

.. autoclass:: synthesized.api.CompressionType
   :members:
