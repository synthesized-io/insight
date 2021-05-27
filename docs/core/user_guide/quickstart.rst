===============
Table Synthesis
===============

Synthesized can be used to quickly generate realistic high dimensional tabular data. This is achieved by learning a high
dimensional, intelligent model of the original data that preserves the statistical properties, distributions and
correlations between fields.

.. important::
    Synthesized assumes that there is no temporal or conditional dependencies the between rows; each row in the table is
    assumed to be independent and identically distributed.

    In addition, maintaining referential links (e.g matching primary keys) across datasets is not currently possible. In
    such a case, the joined dataset must be learned by Synthesized.

.. ipython:: python
    :verbatim:

    import pandas as pd
    import synthesized

.. note::
    Synthesized interfaces with the `pandas <https://pandas.pydata.org/>`_ package, as such the data to be synthesized must
    be first loaded into a DataFrame object.

Load tabular data into a pandas DataFrame

.. ipython:: python
    :verbatim:

    df = pd.read_csv(...)

Extract the ``DataFrameMeta``. This is necessary to learn the schema, which in turn defines the
necessary transformations required to understand the original data and generate new data points.

.. ipython:: python
    :verbatim:

    df_meta = synthesized.MetaExtractor.extract(df)

Initialize the ``HighDimSynthesizer`` object, which will learn an intelligent model of the underlying data.

.. ipython:: python
    :verbatim:

    synth = synthesized.HighDimSynthesizer(df_meta)

Train the model.

.. ipython:: python
    :verbatim:

    synth.learn(df, num_iterations=None)

Synthesize an arbitrary amount of data:

.. ipython:: python
    :verbatim:

    df_synth = synth.synthesize(num_rows=1000)